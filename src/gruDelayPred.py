import pandas               as pd
import numpy                as np
import os
import joblib                
import warnings
import time
import tensorflow           as tf

from sklearn.preprocessing  import LabelEncoder, MinMaxScaler
from sklearn.metrics        import mean_absolute_error, mean_squared_error


from tensorflow.keras.models       import Model
from tensorflow.keras.layers       import (
    Input, Masking, GRU, Dense, Dropout, BatchNormalization,
    Flatten, Attention
)
from tensorflow.keras.callbacks    import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers   import Adam
warnings.filterwarnings("ignore", category=UserWarning,  module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore some TF warnings

# --- CORAL Loss Functions ---
try:
    from coralLoss import coral_loss, levels_from_logits
except ImportError:
    print("coral_loss.py not found. Defining CORAL functions locally.")
    def coral_loss(num_classes):
        def loss(y_true_levels, y_pred_logits):
            y_true_levels           = tf.cast(y_true_levels, tf.int32)
            y_true_levels           = tf.squeeze(y_true_levels)
            val                     = tf.cast(tf.range(0, num_classes - 1), tf.int32)
            val                     = tf.expand_dims(val, axis=0)
            y_true_levels_expanded  = tf.expand_dims(y_true_levels, axis=1)
            y_true_cum              = tf.cast(val < y_true_levels_expanded, tf.float32)
            y_pred_cumprobs         = tf.sigmoid(y_pred_logits)
            log_loss                = - (y_true_cum 
                                      * tf.math.log(y_pred_cumprobs + K.epsilon()) 
                                      + (1.0 - y_true_cum) 
                                      * tf.math.log(1.0 - y_pred_cumprobs + K.epsilon()))
            log_loss                = tf.reduce_sum(log_loss, axis=1)
            log_loss                = tf.reduce_mean(log_loss)
            return log_loss
        return loss

    def levels_from_logits(logits):
        cumprobs       = tf.sigmoid(logits)
        predict_levels = tf.reduce_sum(tf.cast(cumprobs > 0.5, tf.int32), axis=1)
        return predict_levels

# print(f"TensorFlow Version: {tf.__version__}")
# print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- Configuration ---
DATA_DIR         = '../processed_flight_data'
RESULTS_DIR      = '../results_gru_ordinal'
MODEL_DIR        = '../models_gru_ordinal'
TRAIN_FILE       = os.path.join(DATA_DIR, 'train_set.csv')
VALID_FILE       = os.path.join(DATA_DIR, 'validation_set.csv')
TEST_FILE        = os.path.join(DATA_DIR, 'test_set.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# Features
CATEGORICAL_FEATURES = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation', 'Flight_Number']
NUMERICAL_FEATURES   = ['Flight_Duration_Minutes', 'FTD', 'PFD']  # Base numerical features
TARGET_FEATURE       = 'Flight_Delay'

# Ordinal Binning Configuration
DELAY_BINS   = [-float('inf'), 0, 15, float('inf')]
BIN_LABELS   = [0, 1, 2]
NUM_CLASSES  = len(BIN_LABELS)
BIN_MIDPOINTS = {
    0 : 0.0,
    1 : 7.5,
    2 : 45.0
}
print(f"Ordinal classes: {NUM_CLASSES}, Bins: {DELAY_BINS}, Labels: {BIN_LABELS}")
print(f"Bin representative values: {BIN_MIDPOINTS}")

# GRU & Training Configuration
SEQUENCE_LENGTH = 3   # Desired sequence length; groups with fewer rows will be padded
GRU_UNITS       = 128
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 0.001
BATCH_SIZE      = 256
EPOCHS          = 1
PATIENCE        = 10


# --- Data Processor Class ---
class DataProcessor:
    def __init__(self,
                 model_dir       : str,
                 sequence_length : int = SEQUENCE_LENGTH):
        self.model_dir            = model_dir
        self.sequence_length      = sequence_length
        self.categorical_features = CATEGORICAL_FEATURES
        self.numerical_features   = NUMERICAL_FEATURES
        self.target_feature       = TARGET_FEATURE
        self.delay_bins           = DELAY_BINS
        self.bin_labels           = BIN_LABELS
        self.num_classes          = NUM_CLASSES
        self.bin_midpoints        = BIN_MIDPOINTS

    @staticmethod
    def create_sequences_from_df(df, feature_cols, sequence_length, target_level_col, target_delay_col, pad_value=-1):
        """
        Create sliding window sequences from the dataframe grouped by Tail_Number.
        For groups shorter than sequence_length, pad at the beginning with pad_value.
        """
        sequences_X         = []
        sequences_y_levels  = []
        sequences_y_delay   = []
        
        for tail, group in df.groupby('Tail_Number'):
            group         = group.sort_values('Schedule_DateTime')
            X_group       = group[feature_cols].values   # shape: (num_rows, num_features)
            y_levels_group= group[target_level_col].values
            y_delay_group = group[target_delay_col].values
            
            n = len(group)
            if n >= sequence_length:
                # Create sequences using a sliding window
                for i in range(n - sequence_length + 1):
                    sequences_X.append(X_group[i : i + sequence_length])
                    sequences_y_levels.append(y_levels_group[i + sequence_length - 1])
                    sequences_y_delay.append(y_delay_group[i + sequence_length - 1])
            else:
                # Pad the group to create one sequence of fixed length
                padding_needed = sequence_length - n
                padded_sequence= np.pad(X_group, ((padding_needed, 0), (0, 0)),
                                         mode='constant', constant_values=pad_value)
                sequences_X.append(padded_sequence)
                sequences_y_levels.append(y_levels_group[-1])
                sequences_y_delay.append(y_delay_group[-1])
                
        return np.array(sequences_X), np.array(sequences_y_levels), np.array(sequences_y_delay)

    def load_and_preprocess_data(self, file_path, encoders=None, scaler=None, fit_transform=False):
        """Loads data, performs preprocessing, encoding, scaling, and sequence creation."""
        print(f"Loading data from: {file_path}")
        df         = pd.read_csv(file_path)
        start_rows = len(df)
        
        print("Sorting data by Tail_Number and Schedule_DateTime...")
        df['Schedule_DateTime'] = pd.to_datetime(df['Schedule_DateTime'])
        df                      = df.sort_values(by=['Tail_Number', 'Schedule_DateTime']).reset_index(drop=True)
        
        df['Flight_Duration_Minutes'] = df['Flight_Duration_Minutes'].clip(lower=0)
        
        print("Extracting time features...")
        df['Month']                = df['Schedule_DateTime'].dt.month
        df['Day']                  = df['Schedule_DateTime'].dt.day
        df['DayOfWeek']            = df['Schedule_DateTime'].dt.dayofweek
        df['Hour']                 = df['Schedule_DateTime'].dt.hour
        df['Minute']               = df['Schedule_DateTime'].dt.minute
        time_features              = ['Month', 'Day', 'DayOfWeek', 'Hour', 'Minute']
        current_numerical_features = self.numerical_features + time_features
        
        print("Handling missing values...")
        df[current_numerical_features] = df[current_numerical_features].fillna(0)
        df[self.categorical_features]  = df[self.categorical_features].fillna('Unknown')
        df[self.target_feature]        = df[self.target_feature].fillna(0)
        print(f"Target '{self.target_feature}' range before binning: min={df[self.target_feature].min():.2f}, max={df[self.target_feature].max():.2f}")
        
        # Encode Categorical Features
        if fit_transform:
            print("Fitting and transforming categorical features...")
            encoders = {col : LabelEncoder() for col in self.categorical_features}
            for col, encoder in encoders.items():
                unique_values = df[col].astype(str).unique()
                encoder.fit(unique_values)
                df[col] = encoder.transform(df[col].astype(str))
                print(f"  Encoded '{col}' with {len(encoder.classes_)} unique values.")
            joblib.dump(encoders, os.path.join(self.model_dir, 'label_encoders.joblib'))
        else:
            print("Transforming categorical features using loaded encoders...")
            if encoders is None:
                raise ValueError("Encoders must be provided if fit_transform is False")
            for col, encoder in encoders.items():
                known_values = set(encoder.classes_)
                df[col] = df[col].astype(str).apply(lambda s: s if s in known_values else '<unknown>')
                if '<unknown>' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, '<unknown>')
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError as e:
                    print(f"Error transforming column {col}: {e}")
                    df[col] = df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        
        # Scale Numerical Features
        if fit_transform:
            print(f"Fitting and transforming numerical features: {current_numerical_features}")
            scaler = MinMaxScaler()
            df[current_numerical_features] = scaler.fit_transform(df[current_numerical_features])
            joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.joblib'))
            joblib.dump(current_numerical_features, os.path.join(self.model_dir, 'numerical_features_list.joblib'))
        else:
            print("Transforming numerical features using loaded scaler...")
            if scaler is None:
                raise ValueError("Scaler must be provided if fit_transform is False")
            scaled_feature_list = joblib.load(os.path.join(self.model_dir, 'numerical_features_list.joblib'))
            if list(scaled_feature_list) != current_numerical_features:
                print(f"Warning: Feature list mismatch. Scaler trained on {scaled_feature_list}, current features {current_numerical_features}")
            df[current_numerical_features] = scaler.transform(df[current_numerical_features])
        
        print(f"Binning target variable '{self.target_feature}' into {self.num_classes} classes...")
        df[f'{self.target_feature}_Level'] = pd.cut(
            df[self.target_feature],
            bins    = self.delay_bins,
            labels  = self.bin_labels,
            right   = True,
            include_lowest=True
        )
        df[f'{self.target_feature}_Level'] = df[f'{self.target_feature}_Level'].astype(int)
        print(f"Value counts for '{self.target_feature}_Level':\n{df[f'{self.target_feature}_Level'].value_counts(normalize=True)}")
        
        feature_cols = self.categorical_features + current_numerical_features
        X            = df[feature_cols].values
        y_level      = df[f'{self.target_feature}_Level'].values
        y_delay      = df[self.target_feature].values
        
        # Create sequences using sliding-window with padding if sequence length > 1
        if self.sequence_length > 1:
            X, y_level, y_delay = self.create_sequences_from_df(df, feature_cols, self.sequence_length,
                                                                 f'{self.target_feature}_Level', self.target_feature,
                                                                 pad_value=-1)
            print(f"Created sequences: X shape: {X.shape}, y_levels shape: {y_level.shape}")
        else:
            X = X.reshape((X.shape[0], self.sequence_length, X.shape[1]))
            print(f"Reshaped X to: {X.shape}")
        
        final_rows = len(df)
        if start_rows != final_rows:
            print(f"Warning: Row count changed during preprocessing ({start_rows} -> {final_rows})")
        
        return df, X, y_level, y_delay, encoders, scaler, feature_cols

    def predict_ordinal_delay(self, level):
        """Converts predicted level (bin index) to a representative delay value."""
        return self.bin_midpoints.get(level, self.bin_midpoints[max(self.bin_midpoints.keys())])

    @staticmethod
    def calculate_accuracy(y_true_levels, y_pred_levels):
        """Calculates exact match accuracy for ordinal levels."""
        return np.mean(y_true_levels == y_pred_levels)

    def evaluate_model(self, y_true_delay, y_pred_delay, y_true_levels, y_pred_levels):
        """Calculates MAE, RMSE, MSE, and Accuracy."""
        mae      = mean_absolute_error(y_true_delay, y_pred_delay)
        mse      = mean_squared_error(y_true_delay, y_pred_delay)
        rmse     = np.sqrt(mse)
        accuracy = self.calculate_accuracy(y_true_levels, y_pred_levels)
        print(f"  MAE (on delay values): {mae:.4f}")
        print(f"  MSE (on delay values): {mse:.4f}")
        print(f"  RMSE (on delay values): {rmse:.4f}")
        print(f"  Accuracy (Exact Bin Match): {accuracy:.4f}")
        return mae, mse, rmse, accuracy


# --- Model Builder Class ---
class ModelBuilder:
    def __init__(self,
                 input_shape : tuple,
                 num_classes : int,
                 gru_units   : int   = GRU_UNITS,
                 dropout_rate: float = DROPOUT_RATE):
        self.input_shape  = input_shape
        self.num_classes  = num_classes
        self.gru_units    = gru_units
        self.dropout_rate = dropout_rate

    def build_gru_model(self):
        """Builds a standard GRU model for ordinal classification with masking."""
        print("Building Standard GRU Model...")
        inputs = Input(shape=self.input_shape)
        x      = Masking(mask_value=-1)(inputs)
        x      = BatchNormalization()(x)
        x      = GRU(self.gru_units, return_sequences=False, dropout=self.dropout_rate,
                     recurrent_dropout=self.dropout_rate)(x)
        x      = Dense(self.gru_units // 2, activation='relu')(x)
        x      = Dropout(self.dropout_rate)(x)
        outputs= Dense(self.num_classes - 1, activation=None)(x)
        model  = Model(inputs, outputs)
        return model

    def build_gru_attention_model(self):
        """Builds a GRU model with Attention for ordinal classification with masking."""
        print("Building GRU Model with Attention...")
        inputs = Input(shape=self.input_shape)
        x      = Masking(mask_value=-1)(inputs)
        x      = BatchNormalization()(x)
        gru_out= GRU(self.gru_units, return_sequences=True, dropout=self.dropout_rate,
                     recurrent_dropout=self.dropout_rate)(x)
        attention_layer = Attention(use_scale=True)
        attention_out   = attention_layer([gru_out, gru_out])
        context_vector  = Flatten()(attention_out)
        x      = Dense(self.gru_units // 2, activation='relu')(context_vector)
        x      = Dropout(self.dropout_rate)(x)
        outputs= Dense(self.num_classes - 1, activation=None)(x)
        model  = Model(inputs, outputs)
        return model


# --- GRU Pipeline Class ---
class GRUPipeline:
    def __init__(self,
                 data_dir        : str,
                 results_dir     : str,
                 model_dir       : str,
                 sequence_length : int   = SEQUENCE_LENGTH,
                 gru_units       : int   = GRU_UNITS,
                 dropout_rate    : float = DROPOUT_RATE,
                 learning_rate   : float = LEARNING_RATE,
                 batch_size      : int   = BATCH_SIZE,
                 epochs          : int   = EPOCHS,
                 patience        : int   = PATIENCE):
        self.data_dir        = data_dir
        self.results_dir     = results_dir
        self.model_dir       = model_dir
        self.sequence_length = sequence_length
        self.gru_units       = gru_units
        self.dropout_rate    = dropout_rate
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.patience        = patience

        self.train_file = os.path.join(self.data_dir, 'train_set.csv')
        self.val_file   = os.path.join(self.data_dir, 'validation_set.csv')
        self.test_file  = os.path.join(self.data_dir, 'test_set.csv')

        self.data_processor = DataProcessor(self.model_dir, self.sequence_length)
        self.encoders       = None
        self.scaler         = None
        self.df_train       = None
        self.df_val         = None
        self.df_test_orig   = None
        self.X_train        = None
        self.y_train_levels = None
        self.y_train_delay  = None
        self.X_val          = None
        self.y_val_levels   = None
        self.y_val_delay    = None
        self.X_test         = None
        self.y_test_levels  = None
        self.y_test_delay   = None
        self.feature_cols   = None

        self.trained_models    = {}
        self.training_history  = {}
        self.evaluation_result = {}

    def load_data(self):
        print("--- (1/4) Loading and Preprocessing Data ---")
        start_time = time.time()
        (self.df_train, self.X_train, self.y_train_levels, self.y_train_delay,
         self.encoders, self.scaler, self.feature_cols) = self.data_processor.load_and_preprocess_data(
            self.train_file, fit_transform=True
        )
        (self.df_val, self.X_val, self.y_val_levels, self.y_val_delay, _, _, _) = self.data_processor.load_and_preprocess_data(
            self.val_file, encoders=self.encoders, scaler=self.scaler, fit_transform=False
        )
        (self.df_test_orig, self.X_test, self.y_test_levels, self.y_test_delay, _, _, _) = self.data_processor.load_and_preprocess_data(
            self.test_file, encoders=self.encoders, scaler=self.scaler, fit_transform=False
        )
        print(f"Data loading and preprocessing took: {time.time() - start_time:.2f} seconds.")
        print(f"\nInput shape for GRU: {self.X_train.shape[1:]}")
        print(f"Number of classes (delay bins): {self.data_processor.num_classes}")
        print(f"Features used ({len(self.feature_cols)}): {self.feature_cols}")

    def build_models(self):
        input_shape  = (self.X_train.shape[1], self.X_train.shape[2])
        model_builder= ModelBuilder(input_shape, self.data_processor.num_classes, self.gru_units, self.dropout_rate)
        models_to_train = {
            "gru"          : model_builder.build_gru_model(),
            "gru_attention": model_builder.build_gru_attention_model()
        }
        return models_to_train

    def train_models(self, models_to_train):
        for model_name, model in models_to_train.items():
            print(f"\n--- (2/4) Training Model: {model_name} ---")
            model.summary(line_length=120)
            train_start = time.time()
            model.compile(optimizer = Adam(learning_rate=self.learning_rate),
                          loss      = coral_loss(num_classes=self.data_processor.num_classes))
            model_checkpoint_path = os.path.join(self.model_dir, f'{model_name}_best.keras')
            callbacks = [
                tf.keras.callbacks.TerminateOnNaN(),
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=1),
                ModelCheckpoint(filepath=model_checkpoint_path, monitor='val_loss', save_best_only=True,
                                verbose=1, save_weights_only=False),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=self.patience // 2,
                                  min_lr=self.learning_rate / 100, verbose=1)
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
            self.trained_models[model_name]   = model
            print(f"Finished training {model_name}. Time taken: {time.time() - train_start:.2f} seconds.")

    def evaluate_and_save(self):
        print("\n--- (3/4) Predicting and Evaluating on Test Set ---")
        for model_name, model in self.trained_models.items():
            print(f"\n--- Evaluating Model: {model_name} ---")
            predict_start = time.time()
            pred_logits   = model.predict(self.X_test, batch_size=self.batch_size, verbose=1)
            predicted_levels = levels_from_logits(tf.convert_to_tensor(pred_logits)).numpy()
            predicted_delays = [self.data_processor.predict_ordinal_delay(lvl) for lvl in predicted_levels]
            
            mae, mse, rmse, accuracy = self.data_processor.evaluate_model(
                self.y_test_delay,
                np.array(predicted_delays),
                self.y_test_levels,
                predicted_levels
            )
            self.evaluation_result[model_name] = {
                'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'Accuracy': accuracy
            }
            print(f"Evaluation and prediction for {model_name} took: {time.time() - predict_start:.2f} seconds.")
            
            # --- Save Results ---
            print(f"Saving results for {model_name}...")
            results_df = self.df_test_orig.copy()
            # For sequences > 1, save predictions for the last flight of each tail group.
            results_df = results_df.groupby('Tail_Number').tail(1).reset_index(drop=True)
            results_df[f'Predicted_{self.data_processor.target_feature}_Level_{model_name}'] = predicted_levels
            results_df[f'Predicted_{self.data_processor.target_feature}_{model_name}']       = predicted_delays
            output_filename = os.path.join(self.results_dir, f'test_set_with_predictions_{model_name}.csv')
            try:
                results_df.to_csv(output_filename, index=False)
                print(f"Results saved to {output_filename}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")

    def summarize_evaluation(self):
        print("\n--- (4/4) Evaluation Summary ---")
        summary_df = pd.DataFrame(self.evaluation_result).T
        print(summary_df)
        summary_filename = os.path.join(self.results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_filename)
        print(f"Evaluation summary saved to {summary_filename}")

    def run(self):
        self.load_data()
        models_to_train = self.build_models()
        self.train_models(models_to_train)
        self.evaluate_and_save()
        self.summarize_evaluation()
        print("\nPipeline finished.")


if __name__ == '__main__':
    pipeline = GRUPipeline(
        data_dir        = DATA_DIR,
        results_dir     = RESULTS_DIR,
        model_dir       = MODEL_DIR,
        sequence_length = SEQUENCE_LENGTH,
        gru_units       = GRU_UNITS,
        dropout_rate    = DROPOUT_RATE,
        learning_rate   = LEARNING_RATE,
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        patience        = PATIENCE
    )
    pipeline.run()

