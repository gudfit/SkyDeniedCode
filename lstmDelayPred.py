import pandas               as pd
import numpy                as np
import os
import joblib               # For saving/loading scalers/encoders
import warnings

from sklearn.preprocessing   import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error

import tensorflow           as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, TimeDistributed,
    BatchNormalization, Attention
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers  import Adam
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


# --- CORAL Loss functions ---
try:
    from coral_loss import coral_loss, levels_from_logits
except ImportError:
    import tensorflow as tf
    from tensorflow.keras import backend as K

    def coral_loss(num_classes):
        def loss(y_true_levels, y_pred_logits):
            y_true_levels           = tf.cast(y_true_levels, tf.int32)
            y_true_levels           = tf.squeeze(y_true_levels)
            val                     = tf.cast(tf.range(0, num_classes - 1), tf.int32)
            val                     = tf.expand_dims(val, axis=0)
            y_true_levels_expanded  = tf.expand_dims(y_true_levels, axis=1)
            y_true_cum              = tf.cast(val < y_true_levels_expanded, tf.float32)
            y_pred_cumprobs         = tf.sigmoid(y_pred_logits)
            log_loss                = -tf.reduce_sum(
                (y_true_cum * K.log(y_pred_cumprobs + K.epsilon()) +
                 (1.0 - y_true_cum) * K.log(1.0 - y_pred_cumprobs + K.epsilon())),
                axis=1
            )
            return K.mean(log_loss)
        return loss

    def levels_from_logits(logits):
        cumprobs       = tf.sigmoid(logits)
        predict_levels = tf.reduce_sum(tf.cast(cumprobs > 0.5, tf.int32), axis=1)
        return predict_levels


# --- Data Processor Class ---
class DataProcessor:
    def __init__(self,
                 data_dir       : str,
                 model_dir      : str,
                 sequence_length: int = 2):
        self.data_dir             = data_dir
        self.model_dir            = model_dir
        self.sequence_length      = sequence_length
        self.categorical_features = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation', 'Flight_Number']
        self.numerical_features   = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']
        self.target_feature       = 'Flight_Delay'
        self.delay_bins           = [-float('inf'), 0, 15, float('inf')]
        self.bin_labels           = [0, 1, 2]
        self.num_classes          = len(self.bin_labels)
        self.bin_midpoints        = [0, 7.5, 45]

    @staticmethod
    def create_sequences(X, y, sequence_length):
        """
        Create non-overlapping sequences from the 2D arrays X and y.
        This method chops the data into blocks of sequence_length.
        y for each sequence is taken as the label of the last element.
        """
        num_full_sequences = X.shape[0] // sequence_length
        X                  = X[: num_full_sequences * sequence_length]
        y                  = y[: num_full_sequences * sequence_length]
        X_seq              = X.reshape((num_full_sequences, sequence_length, X.shape[1]))
        y_seq              = y[sequence_length - 1 :: sequence_length]
        return X_seq, y_seq

    def load_and_preprocess_data(self,
                                 file_path   : str,
                                 encoders    = None,
                                 scaler      = None,
                                 fit_transform: bool = False):
        """Loads data, performs preprocessing, encoding, scaling, and sorts by flight schedule."""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        # 1. Sort values (by Tail_Number and Schedule_DateTime)
        df['Schedule_DateTime'] = pd.to_datetime(df['Schedule_DateTime'])
        df                    = df.sort_values(by=['Tail_Number', 'Schedule_DateTime']).reset_index(drop=True)

        # Feature Engineering: Extract time features
        df['Month']     = df['Schedule_DateTime'].dt.month
        df['Day']       = df['Schedule_DateTime'].dt.day
        df['DayOfWeek'] = df['Schedule_DateTime'].dt.dayofweek
        df['Hour']      = df['Schedule_DateTime'].dt.hour
        df['Minute']    = df['Schedule_DateTime'].dt.minute

        # Handle missing values
        df[self.numerical_features]   = df[self.numerical_features].fillna(0)
        df[self.categorical_features] = df[self.categorical_features].fillna('Unknown')

        # 2. Encode Categorical Features
        if fit_transform:
            encoders = {col : LabelEncoder() for col in self.categorical_features}
            for col, encoder in encoders.items():
                df[col] = encoder.fit_transform(df[col].astype(str))
            joblib.dump(encoders, os.path.join(self.model_dir, 'label_encoders.joblib'))
        else:
            if encoders is None:
                raise ValueError("Encoders must be provided if fit_transform is False")
            for col, encoder in encoders.items():
                df[col] = df[col].astype(str).map(lambda s: '<unknown>' if s not in encoder.classes_ else s)
                encoder.classes_ = np.append(encoder.classes_, '<unknown>')
                df[col] = encoder.transform(df[col])

        # 3. Scale Numerical Features
        if fit_transform:
            scaler = MinMaxScaler()
            df[self.numerical_features] = scaler.fit_transform(df[self.numerical_features])
            joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided if fit_transform is False")
            df[self.numerical_features] = scaler.transform(df[self.numerical_features])

        # 4. Bin Target Variable (Flight_Delay)
        df[f'{self.target_feature}_Level'] = pd.cut(
            df[self.target_feature],
            bins   = self.delay_bins,
            labels = self.bin_labels,
            right  = True
        ).astype(int)

        feature_cols = self.categorical_features + self.numerical_features
        X            = df[feature_cols].values
        y            = df[f'{self.target_feature}_Level'].values

        if self.sequence_length > 1:
            X, y = self.create_sequences(X, y, self.sequence_length)
        else:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
        return df, X, y, encoders, scaler

    def predict_ordinal_delay(self, level):
        """Convert predicted level (bin index) to a representative delay value."""
        if level < 0 or level >= len(self.bin_midpoints):
            return self.bin_midpoints[-1]
        return self.bin_midpoints[level]

    @staticmethod
    def calculate_accuracy(y_true_levels, y_pred_levels):
        """Calculate exact match accuracy for ordinal levels."""
        return np.mean(y_true_levels == y_pred_levels)

    @staticmethod
    def calculate_per_sample_accuracy(y_true_level, y_pred_level):
        """Calculate accuracy for a single sample."""
        return 1 if y_true_level == y_pred_level else 0

    def evaluate_model(self,
                       y_true_delay : np.ndarray,
                       y_pred_delay : np.ndarray,
                       y_true_levels: np.ndarray,
                       y_pred_levels: np.ndarray):
        """Compute MAE, RMSE, MSE, and Accuracy."""
        mae      = mean_absolute_error(y_true_delay, y_pred_delay)
        mse      = mean_squared_error(y_true_delay, y_pred_delay)
        rmse     = np.sqrt(mse)
        accuracy = self.calculate_accuracy(y_true_levels, y_pred_levels)
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Accuracy (Exact Bin Match): {accuracy:.4f}")
        return mae, mse, rmse, accuracy


# --- Model Builder Class ---
class ModelBuilder:
    def __init__(self,
                 input_shape : tuple,
                 num_classes : int,
                 lstm_units  : int   = 64,
                 dropout_rate: float = 0.2):
        self.input_shape  = input_shape
        self.num_classes  = num_classes
        self.lstm_units   = lstm_units
        self.dropout_rate = dropout_rate

    def build_lstm_model(self):
        """Builds a standard LSTM model for ordinal classification."""
        inputs  = Input(shape=self.input_shape)
        x       = BatchNormalization()(inputs)
        x       = LSTM(self.lstm_units, return_sequences=False, unroll=True)(x)
        x       = Dropout(self.dropout_rate)(x)
        x       = Dense(self.lstm_units // 2, activation='relu')(x)
        outputs = Dense(self.num_classes - 1, activation=None)(x)
        model   = Model(inputs, outputs)
        return model

    def build_lstm_attention_model(self):
        """Builds an LSTM model with an Attention mechanism for ordinal classification."""
        inputs         = Input(shape=self.input_shape)
        x              = BatchNormalization()(inputs)
        lstm_out       = LSTM(self.lstm_units, return_sequences=True, unroll=True)(x)
        lstm_out       = Dropout(self.dropout_rate)(lstm_out)
        attention_out  = Attention()([lstm_out, lstm_out])
        context_vector = tf.keras.layers.Flatten()(attention_out)
        x              = Dense(self.lstm_units // 2, activation='relu')(context_vector)
        x              = Dropout(self.dropout_rate)(x)
        outputs        = Dense(self.num_classes - 1, activation=None)(x)
        model          = Model(inputs, outputs)
        return model


# --- Flight Delay Pipeline Class ---
class FlightDelayPipeline:
    def __init__(self,
                 data_dir        : str,
                 results_dir     : str,
                 model_dir       : str,
                 sequence_length : int   = 2,
                 lstm_units      : int   = 64,
                 dropout_rate    : float = 0.2,
                 learning_rate   : float = 0.001,
                 batch_size      : int   = 128,
                 epochs          : int   = 1,
                 patience        : int   = 10):
        self.data_dir        = data_dir
        self.results_dir     = results_dir
        self.model_dir       = model_dir
        self.sequence_length = sequence_length
        self.lstm_units      = lstm_units
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.patience        = patience

        self.train_file = os.path.join(self.data_dir, 'train_set.csv')
        self.val_file   = os.path.join(self.data_dir, 'validation_set.csv')
        self.test_file  = os.path.join(self.data_dir, 'test_set.csv')

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir,   exist_ok=True)

        self.data_processor      = DataProcessor(self.data_dir, self.model_dir, self.sequence_length)
        self.encoders            = None
        self.scaler              = None
        self.X_train             = None
        self.y_train_levels      = None
        self.X_val               = None
        self.y_val_levels        = None
        self.df_test_orig        = None
        self.X_test_orig         = None
        self.y_test_levels_orig  = None

        self.trained_models      = {}
        self.training_history    = {}

    def load_data(self):
        print("--- Preprocessing Training Data ---")
        df_train, X_train, y_train_levels, self.encoders, self.scaler = self.data_processor.load_and_preprocess_data(
            self.train_file, fit_transform=True
        )
        print("\n--- Preprocessing Validation Data ---")
        df_val, X_val, y_val_levels, _, _ = self.data_processor.load_and_preprocess_data(
            self.val_file, encoders=self.encoders, scaler=self.scaler, fit_transform=False
        )
        print("\n--- Preprocessing Test Data ---")
        df_test_orig, X_test_orig, y_test_levels_orig, _, _ = self.data_processor.load_and_preprocess_data(
            self.test_file, encoders=self.encoders, scaler=self.scaler, fit_transform=False
        )
        self.X_train            = X_train
        self.y_train_levels     = y_train_levels
        self.X_val              = X_val
        self.y_val_levels       = y_val_levels
        self.df_test_orig       = df_test_orig
        self.X_test_orig        = X_test_orig
        self.y_test_levels_orig = y_test_levels_orig

    def build_models(self):
        input_shape   = (self.X_train.shape[1], self.X_train.shape[2])
        print(f"\nInput shape for LSTM: {input_shape}")
        print(f"Number of classes (delay bins): {self.data_processor.num_classes}")
        model_builder = ModelBuilder(input_shape,
                                     self.data_processor.num_classes,
                                     self.lstm_units,
                                     self.dropout_rate)
        models_to_train = {
            "lstm"          : model_builder.build_lstm_model(),
            "lstm_attention": model_builder.build_lstm_attention_model()
        }
        return models_to_train

    def train_models(self, models_to_train):
        for model_name, model in models_to_train.items():
            print(f"\n--- Training Model: {model_name} ---")
            model.summary()
            model.compile(
                optimizer = Adam(learning_rate=self.learning_rate),
                loss      = coral_loss(num_classes=self.data_processor.num_classes)
            )
            model_checkpoint_path = os.path.join(self.model_dir, f'{model_name}_best.keras')
            callbacks             = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=self.patience // 2, min_lr=self.learning_rate / 100, verbose=1)
            ]
            history = model.fit(
                self.X_train, self.y_train_levels,
                validation_data = (self.X_val, self.y_val_levels),
                epochs          = self.epochs,
                batch_size      = self.batch_size,
                callbacks       = callbacks,
                verbose         = 1
            )
            self.training_history[model_name] = history
            model.load_weights(model_checkpoint_path)
            self.trained_models[model_name]   = model
            print(f"Finished training {model_name}.")

    def iterative_prediction(self, model, model_name):
        # Get indices for 'PFD' and 'Tail_Number'
        try:
            pfd_index = self.data_processor.numerical_features.index('PFD')
            tn_index  = self.data_processor.categorical_features.index('Tail_Number')
            print(f"Index for PFD feature (in numerical features): {pfd_index}")
            print(f"Index for Tail_Number feature (in categorical features): {tn_index}")
        except ValueError as e:
            print(f"Error finding feature index: {e}")
            exit()

        X_test_iterative      = np.copy(self.X_test_orig)
        predicted_levels_list = []
        predicted_delays_list = []
        accuracy_list         = []
        seq_tail_numbers      = X_test_iterative[:, -1, tn_index]

        print("Starting iterative prediction loop...")
        for i in range(len(X_test_iterative)):
            current_input   = X_test_iterative[i : i+1, :, :]
            pred_logits     = model.predict(current_input, verbose=0)[0]
            predicted_level = levels_from_logits(tf.expand_dims(pred_logits, axis=0)).numpy()[0]
            predicted_levels_list.append(predicted_level)
            predicted_delay  = self.data_processor.predict_ordinal_delay(predicted_level)
            predicted_delays_list.append(predicted_delay)
            true_level      = self.y_test_levels_orig[i]
            sample_accuracy = self.data_processor.calculate_per_sample_accuracy(true_level, predicted_level)
            accuracy_list.append(sample_accuracy)

            if i + 1 < len(X_test_iterative):
                if seq_tail_numbers[i] == X_test_iterative[i+1, 0, tn_index]:
                    try:
                        num_start                      = len(self.data_processor.categorical_features)
                        original_scaled_numerical_next = X_test_iterative[i+1, 0, num_start:]
                        temp_full_numerical_next       = original_scaled_numerical_next.reshape(1, -1)
                        original_unscaled_numerical_next = self.scaler.inverse_transform(temp_full_numerical_next)
                        original_unscaled_numerical_next[0, pfd_index] = predicted_delay
                        updated_scaled_numerical_next  = self.scaler.transform(original_unscaled_numerical_next)
                        scaled_predicted_pfd           = updated_scaled_numerical_next[0, pfd_index]
                        X_test_iterative[i+1, 0, num_start + pfd_index] = scaled_predicted_pfd
                    except Exception as e:
                        print(f"  Warning: Could not update PFD for sequence {i+1}: {e}")
        print("Finished iterative prediction loop.")
        return predicted_levels_list, predicted_delays_list, accuracy_list

    def evaluate_and_save(self):
        for model_name, model in self.trained_models.items():
            print(f"\n--- Evaluating Model: {model_name} on Test Set ---")
            predicted_levels_list, predicted_delays_list, accuracy_list = self.iterative_prediction(model, model_name)
            y_true_delay_test = self.df_test_orig[self.data_processor.target_feature].values[:len(predicted_delays_list)]
            self.data_processor.evaluate_model(
                y_true_delay_test,
                np.array(predicted_delays_list),
                self.y_test_levels_orig,
                np.array(predicted_levels_list)
            )
            results_df = self.df_test_orig.iloc[
                self.sequence_length - 1 : len(predicted_delays_list) * self.sequence_length : self.sequence_length
            ].copy()
            results_df[f'Predicted_{self.data_processor.target_feature}_Level_{model_name}'] = predicted_levels_list
            results_df[f'Predicted_{self.data_processor.target_feature}_{model_name}']       = predicted_delays_list
            results_df[f'Prediction_Accuracy_{model_name}']                                  = accuracy_list

            output_filename = os.path.join(self.results_dir, f'test_set_with_predictions_{model_name}.csv')
            results_df.to_csv(output_filename, index=False)
            print(f"Results saved to {output_filename}")

    def run(self):
        self.load_data()
        models_to_train = self.build_models()
        self.train_models(models_to_train)
        self.evaluate_and_save()
        print("\nPipeline finished.")


if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR        = 'processed_flight_data'
    RESULTS_DIR     = 'results_lstm_ordinal'
    MODEL_DIR       = 'models_lstm_ordinal'
    SEQUENCE_LENGTH = 2
    LSTM_UNITS      = 64
    DROPOUT_RATE    = 0.2
    LEARNING_RATE   = 0.001
    BATCH_SIZE      = 128
    EPOCHS          = 1
    PATIENCE        = 10

    pipeline = FlightDelayPipeline(
        data_dir        = DATA_DIR,
        results_dir     = RESULTS_DIR,
        model_dir       = MODEL_DIR,
        sequence_length = SEQUENCE_LENGTH,
        lstm_units      = LSTM_UNITS,
        dropout_rate    = DROPOUT_RATE,
        learning_rate   = LEARNING_RATE,
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        patience        = PATIENCE
    )
    pipeline.run()

