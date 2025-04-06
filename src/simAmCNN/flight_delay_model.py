import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import joblib # For saving/loading preprocessor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential future pandas warnings

# ================ DATA PREPROCESSING MODULE ================

class FlightDataPreprocessor:
    """
    Handles preprocessing of flight data for delay prediction model.
    Handles fitting on training data and transforming new data.
    """
    def __init__(self, categorical_features, numerical_features, temporal_features):
        """
        Initialize the preprocessor with feature definitions.

        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            temporal_features: List of temporal feature names
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.temporal_features = temporal_features

        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.fitted_columns = None # To ensure transform uses same columns as fit
        self.total_dim = 0
        self.is_fitted = False

    def _extract_temporal_features(self, dt_series):
        """Helper to extract sin/cos features from datetime series."""
        features = {}
        # Hour
        hour = dt_series.dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        # Day of week
        dow = dt_series.dt.dayofweek
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        # Month
        month = dt_series.dt.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        return pd.DataFrame(features, index=dt_series.index)

    def fit(self, df):
        """
        Fit preprocessors on the training data.

        Args:
            df: DataFrame containing the training data
        """
        print("Fitting preprocessor...")
        # Make a copy to avoid modifying original df
        df_processed = df.copy()

        # Handle missing numerical values before scaling
        for feature in self.numerical_features:
             # Simple mean imputation for fitting scaler
            mean_val = df_processed[feature].mean()
            df_processed[feature] = df_processed[feature].fillna(mean_val)

        # Fit encoders for categorical features
        categorical_dfs = []
        for feature in self.categorical_features:
            # Handle potential NaN values before encoding
            df_processed[feature] = df_processed[feature].fillna('Missing') # Treat NaN as a category
            unique_values = df_processed[feature].unique()
            # print(f"Fitting OHE for {feature} with {len(unique_values)} unique values.") # Debug
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[np.sort(unique_values)])
            except TypeError:
                # Fall back to older parameter name for scikit-learn < 1.2
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories=[np.sort(unique_values)])

            # Fit and transform the training data column
            encoded_data = encoder.fit_transform(df_processed[[feature]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([feature]), index=df_processed.index)
            categorical_dfs.append(encoded_df)
            self.categorical_encoders[feature] = encoder

        # Fit scaler for numerical features
        numerical_scaled_data = None
        if self.numerical_features:
            # print(f"Fitting Scaler for {self.numerical_features}") # Debug
            numerical_scaled_data = self.numerical_scaler.fit_transform(df_processed[self.numerical_features])
            numerical_df = pd.DataFrame(numerical_scaled_data, columns=self.numerical_features, index=df_processed.index)
        else:
            numerical_df = pd.DataFrame(index=df_processed.index) # Empty df if no numerical features

        # Process temporal features
        temporal_dfs = []
        for feature in self.temporal_features:
            if feature == 'Schedule_DateTime':
                # print(f"Processing temporal features for {feature}") # Debug
                dt_series = pd.to_datetime(df_processed[feature], errors='coerce').fillna(pd.Timestamp('2000-01-01')) # Handle errors/NaN
                temp_df = self._extract_temporal_features(dt_series)
                temporal_dfs.append(temp_df)
            # Add logic for other temporal features if needed

        # Combine all processed features
        final_df = pd.concat([pd.concat(categorical_dfs, axis=1), numerical_df, pd.concat(temporal_dfs, axis=1)], axis=1)

        self.fitted_columns = final_df.columns.tolist()
        self.total_dim = len(self.fitted_columns)
        self.is_fitted = True
        print(f"Preprocessor fitted. Total feature dimension: {self.total_dim}")
        return self

    def transform_single_flight(self, flight_data):
        """
        Transform a single flight's data into a feature vector using fitted processors.

        Args:
            flight_data: Series or dict containing a single flight's data

        Returns:
            Numpy array of processed features
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        # Convert dict to Series if necessary
        # if isinstance(flight_data, dict):
        #     flight_data = pd.Series(flight_data)
        if not isinstance(flight_data, dict):
             try:
                 flight_data = flight_data.to_dict()
             except AttributeError:
                  raise TypeError(f"transform_single_flight expects dict or pandas Series/Row, got {type(flight_data)}")

        features = []

        # Process categorical features
        categorical_vector = []
        for feature in self.categorical_features:
            value = flight_data.get(feature, 'Missing') # Use 'Missing' if feature absent
            if feature in self.categorical_encoders:
                encoder = self.categorical_encoders[feature]
                # Reshape for encoder (needs 2D) and transform
                one_hot = encoder.transform(np.array([[value]]))
                 # Convert to dense if it's sparse (shouldn't be with sparse_output=False)
                if hasattr(one_hot, 'toarray'):
                    one_hot = one_hot.toarray()
                categorical_vector.extend(one_hot.flatten())
            else:
                 # Should not happen if fit was correct, but handle defensively
                 # Need to add placeholder dimensions matching the fitted encoder
                 print(f"Warning: Encoder not found for feature '{feature}' during transform.")
                 # This part is tricky - ideally you know the expected dim from fit
                 # For now, add zeros, but this might be incorrect if fit failed.
                 # A better approach might be to store expected dims during fit.
                 # Let's assume fit worked and use encoder info if available indirectly
                 # Or rely on the final alignment step
                 pass # Rely on final alignment

        # Process numerical features
        numerical_vector = []
        if self.numerical_features:
            numerical_values = []
            for f in self.numerical_features:
                # Use mean from scaler if value is missing/NaN
                mean_val = self.numerical_scaler.mean_[self.numerical_features.index(f)]
                numerical_values.append(flight_data.get(f, mean_val)) # Use mean for missing

            numerical_values = np.array(numerical_values).reshape(1, -1)
            # Handle potential NaN again before transform (should be caught above, but safety)
            numerical_values = np.nan_to_num(numerical_values, nan=np.mean(numerical_values)) # Replace any leftover NaN
            scaled_values = self.numerical_scaler.transform(numerical_values).flatten()
            numerical_vector.extend(scaled_values)

        # Process temporal features
        temporal_vector = []
        for feature in self.temporal_features:
            if feature == 'Schedule_DateTime':
                raw_dt_value = flight_data.get(feature) # Get raw value (could be string, timestamp, None)

                # Attempt conversion, coerce errors to NaT (Not a Time)
                dt = pd.to_datetime(raw_dt_value, errors='coerce')

                # Check if the conversion resulted in NaT (missing/invalid)
                if pd.isna(dt): # pd.isna() correctly handles NaT, None, NaN
                    dt = pd.Timestamp('2000-01-01') # Assign the default timestamp

                # Now 'dt' is guaranteed to be a valid Timestamp object
                # --- END CORRECTION ---

                # Proceed with feature extraction using the valid 'dt'
                dt_series = pd.Series([dt]) # Helper function needs a Series/DatetimeIndex
                temp_df = self._extract_temporal_features(dt_series)
                temporal_vector.extend(temp_df.values.flatten())           # Add logic for other temporal features

        # Combine vectors (order matters!) - build a temporary dict/Series first
        temp_feature_dict = {}

        # Categorical (using feature names from encoder)
        current_idx = 0
        for feature in self.categorical_features:
            if feature in self.categorical_encoders:
                encoder = self.categorical_encoders[feature]
                f_names = encoder.get_feature_names_out([feature])
                num_encoded_features = len(f_names)
                for i, f_name in enumerate(f_names):
                    if current_idx + i < len(categorical_vector):
                         temp_feature_dict[f_name] = categorical_vector[current_idx + i]
                    else: # Handle mismatch if transform failed previously
                         temp_feature_dict[f_name] = 0.0
                current_idx += num_encoded_features

        # Numerical
        for i, feature in enumerate(self.numerical_features):
            if i < len(numerical_vector):
                temp_feature_dict[feature] = numerical_vector[i]
            else:
                temp_feature_dict[feature] = 0.0 # Handle mismatch

        # Temporal (using generated names)
        temp_temporal_vector = []
        for feature in self.temporal_features:
             if feature == 'Schedule_DateTime':
                  base_names = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
                  for i, bn in enumerate(base_names):
                      if i < len(temporal_vector):
                           temp_feature_dict[bn] = temporal_vector[i]
                      else:
                           temp_feature_dict[bn] = 0.0 # Handle mismatch

        # Create final vector aligned with fitted columns
        final_vector = pd.Series(temp_feature_dict).reindex(self.fitted_columns).fillna(0.0).astype(np.float32).values
        # print(f"Single transform shape: {final_vector.shape}") # Debug

        # Ensure final vector has the expected dimension
        if len(final_vector) != self.total_dim:
             print(f"Warning: Dimension mismatch in transform_single_flight. Expected {self.total_dim}, got {len(final_vector)}. Padding/truncating.")
             # Pad or truncate (less ideal, suggests issue in alignment)
             if len(final_vector) < self.total_dim:
                 final_vector = np.pad(final_vector, (0, self.total_dim - len(final_vector)), 'constant')
             else:
                 final_vector = final_vector[:self.total_dim]


        return final_vector


    def transform_flight_chain(self, flight_chain):
        """
        Transform a sequence of flights into a matrix of features using fitted processors.

        Args:
            flight_chain: List of dicts or DataFrame rows containing flight chain data

        Returns:
            Numpy array matrix of shape (T, D) where T is chain length and D is feature dimension
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")

        T = len(flight_chain)
        if T == 0:
             # Return an empty array with the correct feature dimension
             return np.zeros((0, self.total_dim))

        X = np.zeros((T, self.total_dim))

        for t, flight in enumerate(flight_chain):
            try:
                X[t] = self.transform_single_flight(flight)
            except Exception as e:
                print(f"Error transforming flight at index {t}: {flight}")
                print(f"Error message: {e}")
                # Option: Fill with zeros or skip, depends on strategy
                X[t] = np.zeros(self.total_dim) # Fill with zeros for now

        # print(f"Chain transform shape: {X.shape}") # Debug
        return X

    def save(self, filepath):
        """Saves the fitted preprocessor state."""
        state = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'temporal_features': self.temporal_features,
            'categorical_encoders': self.categorical_encoders,
            'numerical_scaler': self.numerical_scaler,
            'fitted_columns': self.fitted_columns,
            'total_dim': self.total_dim,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads a fitted preprocessor state."""
        state = joblib.load(filepath)
        preprocessor = cls(
            state['categorical_features'],
            state['numerical_features'],
            state['temporal_features']
        )
        preprocessor.categorical_encoders = state['categorical_encoders']
        preprocessor.numerical_scaler = state['numerical_scaler']
        preprocessor.fitted_columns = state['fitted_columns']
        preprocessor.total_dim = state['total_dim']
        preprocessor.is_fitted = state['is_fitted']
        print(f"Preprocessor loaded from {filepath}. Total dim: {preprocessor.total_dim}")
        return preprocessor


# ================ FLIGHT CHAIN DATASET ================

class FlightChainDataset(Dataset):
    """
    Dataset for flight chains for PyTorch training.
    Uses a pre-fitted preprocessor.
    """
    def __init__(self, data_path, preprocessor, chain_length=3):
        """
        Args:
            data_path: Path to the CSV file containing flight data
            preprocessor: A *fitted* FlightDataPreprocessor instance
            chain_length: Number of consecutive flights in a chain (T)
        """
        if not preprocessor.is_fitted:
            raise ValueError("Preprocessor must be fitted before creating the dataset.")

        self.data_path = data_path
        self.preprocessor = preprocessor
        self.chain_length = chain_length
        self.total_dim = preprocessor.total_dim # Get dim from preprocessor

        # Load data
        print(f"Loading data from {data_path}...")
        try:
            # Try reading with fallback encoding if default utf-8 fails
            self.df = pd.read_csv(data_path)
        except UnicodeDecodeError:
             print("UTF-8 failed, trying latin1 encoding...")
             self.df = pd.read_csv(data_path, encoding='latin1')
        except FileNotFoundError:
             print(f"Error: Data file not found at {data_path}")
             raise
        print(f"Loaded {len(self.df)} flights.")

        # Ensure necessary columns exist
        required_cols = ['Tail_Number', 'Schedule_DateTime', 'Flight_Duration_Minutes', 'Flight_Delay'] \
                        + preprocessor.categorical_features \
                        + preprocessor.numerical_features
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")

        # Pre-convert datetime to avoid repeated conversions
        self.df['Schedule_DateTime'] = pd.to_datetime(self.df['Schedule_DateTime'], errors='coerce')
        # Drop rows where key sorting/chaining columns are invalid
        self.df.dropna(subset=['Tail_Number', 'Schedule_DateTime', 'Flight_Duration_Minutes', 'Flight_Delay'], inplace=True)


        # Group flights by tail number to form chains
        self.flight_chains = []
        self.delay_labels = []

        print("Grouping flights and creating chains...")
        grouped = self.df.groupby('Tail_Number')
        for tail_number, group in tqdm(grouped, desc="Processing Tail Numbers"):
            # Ensure group has enough flights and sort
            if len(group) < chain_length:
                continue
            sorted_flights = group.sort_values('Schedule_DateTime')

            # Create chains of consecutive flights
            for i in range(len(sorted_flights) - chain_length + 1):
                chain_df = sorted_flights.iloc[i : i + chain_length]

                # --- Chain Validation (Optional but Recommended) ---
                is_valid_chain = True
                # for j in range(1, len(chain_df)):
                #     # Use actual arrival if available, else estimate
                #     # This part depends heavily on available data. Using scheduled for now.
                #     prev_sched_arr = chain_df.iloc[j-1]['Schedule_DateTime'] + pd.Timedelta(minutes=chain_df.iloc[j-1]['Flight_Duration_Minutes'])
                #     curr_sched_dep = chain_df.iloc[j]['Schedule_DateTime']

                #     time_diff_hours = (curr_sched_dep - prev_sched_arr).total_seconds() / 3600

                #     # Check for reasonable ground time (e.g., > 20 mins and < 24 hours)
                #     if not (0.33 < time_diff_hours < 24):
                #         is_valid_chain = False
                #         break
                # --- End Chain Validation ---

                if is_valid_chain:
                    # Extract delay label for the last flight (T-th flight)
                    last_flight_delay = chain_df.iloc[-1]['Flight_Delay']
                    delay_level = self.get_delay_level(last_flight_delay)

                    # Store chain as list of dicts for preprocessor
                    self.flight_chains.append(chain_df.to_dict('records'))
                    self.delay_labels.append(delay_level)

        print(f"Created {len(self.flight_chains)} flight chains.")
        if not self.flight_chains:
            print("Warning: No flight chains were created. Check data and chain_length.")


    def get_delay_level(self, delay_minutes):
        """
        Convert delay in minutes to categorical delay level.

        Args:
            delay_minutes: Flight delay in minutes

        Returns:
            Integer delay level (0-4)
        """
        if delay_minutes <= 0:      # Early or On Time
            return 0
        elif delay_minutes <= 15:   # Slight Delay (<= 15 min)
            return 1
        elif delay_minutes <= 45:   # Moderate Delay (15 < delay <= 45 min)
            return 2
        elif delay_minutes <= 120:  # Significant Delay (45 < delay <= 120 min)
            return 3
        else:                       # Severe Delay (> 120 min)
            return 4


    def __len__(self):
        return len(self.flight_chains)

    def __getitem__(self, idx):
        if idx >= len(self.flight_chains):
             raise IndexError("Index out of range")

        chain = self.flight_chains[idx]
        label = self.delay_labels[idx]

        # Transform the flight chain to matrix X using the preprocessor
        X = self.preprocessor.transform_flight_chain(chain)

        # Ensure the returned tensor dimensions are correct
        if X.shape != (self.chain_length, self.total_dim):
             print(f"Warning: Unexpected shape in __getitem__ at index {idx}. Expected {(self.chain_length, self.total_dim)}, got {X.shape}. Skipping?")
             # This indicates a problem upstream. Maybe return a dummy?
             # Returning dummy data to avoid crashing dataloader
             X = np.zeros((self.chain_length, self.total_dim))
             label = 0 # Assign a default label


        return torch.FloatTensor(X), torch.LongTensor([label]) # Label needs to be in a tensor/list for batching


# ================ MODEL COMPONENTS ================

class SimAM(nn.Module):
    """
    Efficient Implementation of SimAM Attention Module (works with 4D tensors).
    Based on common implementations derived from the paper.
    NOTE: Consider the lambda value carefully.
    """
    def __init__(self, lambda_val=1e-4): # Paper often cited with 1e-4
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: [B, C, H, W] (in your case, H=T, W=D)
        # Make sure input is 4D
        if x.dim() != 4:
            raise ValueError(f"SimAM expects 4D input (B, C, H, W), but got {x.dim()}D shape: {x.shape}")

        b, c, h, w = x.size()
        n = w * h - 1 # Number of elements excluding the target

        # Calculate mean and variance over spatial dimensions (H, W)
        # Keepdims=True is important for broadcasting
        mu = x.mean(dim=[2, 3], keepdim=True)

        # Calculate squared difference from mean
        x_minus_mu_sq = (x - mu).pow(2)

        # Calculate variance estimate (using sum and n)
        # Add small epsilon for numerical stability if variance is near zero
        var_est = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / (n + 1e-8) # Avoid division by zero if n=0 (e.g., 1x1 spatial)


        # Calculate inverse energy E_inv = (x - mu)^2 / (4 * (var + lambda)) + 0.5
        denominator = 4 * (var_est + self.lambda_val)
        e_inv = x_minus_mu_sq / (denominator + 1e-8) + 0.5 # Add epsilon to denominator too

        # Apply sigmoid and multiply with input feature map
        attention_weights = self.sigmoid(e_inv) # Sigmoid maps E_inv to (0, 1)

        return x * attention_weights

class CNNBlock(nn.Module):
    """
    CNN Block potentially with SimAM attention.
    Uses Conv2d, suitable for 4D input [B, C, H, W].
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[(1, 1)], use_simam=True, lambda_val=1e-4): # Default is list of tuples
        super(CNNBlock, self).__init__()
        self.use_simam = use_simam

        layers = []
        current_channels = in_channels
        num_convs = len(kernel_sizes)

        for i, k_size in enumerate(kernel_sizes): # k_size is now expected to be a tuple, e.g., (kh, kw)
            if not isinstance(k_size, tuple) or len(k_size) != 2:
                 raise ValueError(f"CNNBlock expects kernel_sizes to be a list of 2D tuples, got: {k_size}")

            kh, kw = k_size # Unpack kernel height and width

            # --- CORRECTED PADDING CALCULATION for 2D ---
            # Calculate padding for height and width separately
            # Assumes 'same' padding desired for odd kernel dimensions
            padding_h = (kh - 1) // 2
            padding_w = (kw - 1) // 2
            padding = (padding_h, padding_w) # Conv2d padding can be a tuple (padH, padW)
            # --- END CORRECTION ---

            block_out_channels = out_channels # Output channels for this block's final conv

            conv_layer = nn.Conv2d(current_channels, block_out_channels, kernel_size=k_size, padding=padding)
            norm_layer = nn.BatchNorm2d(block_out_channels)
            relu_layer = nn.ReLU()

            layers.extend([conv_layer, norm_layer, relu_layer])
            current_channels = block_out_channels

        self.conv_layers = nn.Sequential(*layers)

        if use_simam:
            self.simam = SimAM(lambda_val=lambda_val)

    def forward(self, x):
        x = self.conv_layers(x)
        if self.use_simam:
            x = self.simam(x)
        return x

class MogrifierLSTMCell(nn.Module):
    """
    MogrifierLSTM Cell as described in the paper.
    """
    def __init__(self, input_size, hidden_size, num_rounds=5): # 5 rounds often cited
        super(MogrifierLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rounds = num_rounds

        # Mogrifier matrices: Q updates x, R updates h
        # Ensure correct dimensions: Q maps h_dim -> x_dim, R maps x_dim -> h_dim
        self.Q_matrices = nn.ModuleList([
            nn.Linear(hidden_size, input_size)
            for _ in range((num_rounds + 1) // 2) # Rounds 1, 3, 5...
        ])

        self.R_matrices = nn.ModuleList([
            nn.Linear(input_size, hidden_size)
            for _ in range(num_rounds // 2)      # Rounds 2, 4, 6...
        ])

        # Standard LSTM cell (takes mogrified input and hidden state)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

    def mogrify(self, x, h):
        """
        Apply Mogrifier interaction between input and hidden state.

        Args:
            x: Input tensor [batch_size, input_size]
            h: Hidden state tensor [batch_size, hidden_size]

        Returns:
            Modified x and h tensors
        """
        x_mod, h_mod = x, h

        # Iterate through mogrifier rounds
        for i in range(1, self.num_rounds + 1):
            if i % 2 == 1:  # Odd rounds (1, 3, ...): update x using h
                q_idx = i // 2
                x_mod = 2 * torch.sigmoid(self.Q_matrices[q_idx](h_mod)) * x_mod
            else:           # Even rounds (2, 4, ...): update h using x
                r_idx = (i // 2) - 1
                h_mod = 2 * torch.sigmoid(self.R_matrices[r_idx](x_mod)) * h_mod

        return x_mod, h_mod

    def forward(self, x, states):
        """
        Forward pass for one time step.

        Args:
            x: Input for the current time step [batch_size, input_size]
            states: Tuple (h, c) of hidden and cell states from previous step
                    h: [batch_size, hidden_size]
                    c: [batch_size, hidden_size]

        Returns:
            Tuple (h_next, c_next) for the current time step
        """
        h, c = states

        # Apply Mogrifier interaction
        x_mod, h_mod = self.mogrify(x, h)

        # Apply standard LSTM cell with modified inputs/states
        # Note: LSTMCell expects (input, (h, c))
        h_next, c_next = self.lstm_cell(x_mod, (h_mod, c))

        return h_next, c_next


class SimAMCNNMogrifierLSTM(nn.Module):
    """
    Combined SimAM-CNN-MogrifierLSTM model for flight delay prediction.
    """
    def __init__(self, input_feature_dim, lstm_hidden_size, num_classes=5,
                 cnn_channels=[32, 64], kernel_sizes=[(1,1), (3,3)], # Use 2D kernels
                 use_simam=True, lambda_val=1e-4, mogrifier_rounds=5,
                 chain_length=3):
        super(SimAMCNNMogrifierLSTM, self).__init__()

        # ... (keep other attributes) ...
        self.input_feature_dim = input_feature_dim # D
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        self.chain_length = chain_length # T

        # CNN blocks with SimAM attention
        self.cnn_blocks = nn.ModuleList()
        prev_channels = 1

        print("Initializing CNN Blocks:")
        for i, out_channels in enumerate(cnn_channels):
            # Get the kernel size for this block (should be a tuple like (kh, kw))
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else (3,3) # Default if not specified
            if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
                 raise ValueError(f"Kernel size must be a 2D tuple (height, width), got: {kernel_size}")

            print(f"  Block {i+1}: In={prev_channels}, Out={out_channels}, Kernel={kernel_size}")

            # --- CORRECTED CALL ---
            # Pass the kernel_size tuple directly to CNNBlock's kernel_sizes argument
            # CNNBlock expects a list/tuple of kernel sizes for its internal loop
            block = CNNBlock(
                in_channels=prev_channels,
                out_channels=out_channels,
                kernel_sizes=[kernel_size], # Pass as a list containing the single kernel tuple
                use_simam=use_simam,
                lambda_val=lambda_val
            )
            # --- END CORRECTION ---

            self.cnn_blocks.append(block)
            prev_channels = out_channels

        # ... (rest of the __init__, including LSTM input size calculation, etc.) ...
        self.cnn_output_channels = cnn_channels[-1]
        self.lstm_input_size = self.cnn_output_channels * self.input_feature_dim
        print(f"LSTM input size calculated: {self.cnn_output_channels} channels * {self.input_feature_dim} features = {self.lstm_input_size}")
        self.mogrifier_lstm = MogrifierLSTMCell(self.lstm_input_size, lstm_hidden_size, mogrifier_rounds)
        self.fc_out = nn.Linear(lstm_hidden_size, num_classes)
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, T, D]
                where T is the chain length and D is feature dimension
        """
        batch_size, T, D = x.size()
        if T != self.chain_length or D != self.input_feature_dim:
             print(f"Warning: Input tensor shape mismatch in forward. Expected [B, {self.chain_length}, {self.input_feature_dim}], got {x.shape}")
             # Might need padding/truncating here if necessary, or raise error

        # Reshape input for CNN: [batch_size, channels=1, T, D]
        # Conv2d expects [B, C_in, H, W]. Here H=T, W=D.
        x_cnn = x.unsqueeze(1) # Add channel dimension

        # Apply CNN blocks
        # Input: [B, 1, T, D]
        for i, block in enumerate(self.cnn_blocks):
            x_cnn = block(x_cnn)
            # print(f" Shape after CNN block {i+1}: {x_cnn.shape}") # Debug
        # Output: [B, C_final, T_out, D_out]
        # Assuming same padding: [B, C_final, T, D]

        # Reshape for LSTM: Need [batch_size, sequence_length, features_per_step]
        # Sequence length is T. Features per step = C_final * D.
        # Output of CNN: [B, C_final, T, D] -> Need [B, T, C_final * D]
        _, C_out, T_out, D_out = x_cnn.shape

        # Permute to bring Time dimension before Channel/Feature dimensions: [B, T, C, D]
        x_permuted = x_cnn.permute(0, 2, 1, 3)

        # Reshape/Flatten the last two dimensions (C_out * D_out)
        x_lstm_input = x_permuted.reshape(batch_size, T_out, C_out * D_out)
        # print(f" Shape after reshape for LSTM: {x_lstm_input.shape}") # Debug

        # Verify LSTM input size matches expectation
        if x_lstm_input.shape[2] != self.lstm_input_size:
            raise ValueError(f"LSTM input dimension mismatch. Expected {self.lstm_input_size}, got {x_lstm_input.shape[2]}")

        # Initialize LSTM states
        h = torch.zeros(batch_size, self.lstm_hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.lstm_hidden_size, device=x.device)

        # Process sequence with MogrifierLSTM step-by-step
        for t in range(T_out): # Iterate over time steps
            # Input for one time step: [batch_size, features_per_step]
            current_input = x_lstm_input[:, t, :]
            h, c = self.mogrifier_lstm(current_input, (h, c))

        # Final prediction using the *last* hidden state 'h'
        out = self.fc_out(h)
        # print(f" Shape of final output: {out.shape}") # Debug

        return out


# ================ TRAINING AND EVALUATION ================

def train_model(model, train_loader, val_loader,
                criterion, optimizer, num_epochs, device,
                model_save_path='best_model.pth', patience=10):
    """
    Train the model with validation and early stopping.
    """
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\nStarting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device).squeeze(1) # Labels [B, 1] -> [B]

            if labels.ndim == 0: # Handle batch size of 1 if squeeze removes all dims
                 labels = labels.unsqueeze(0)
            if inputs.ndim == 2: # Handle batch size of 1 for inputs
                 inputs = inputs.unsqueeze(0)

            if inputs.shape[0] != labels.shape[0]:
                 print(f"Skipping batch due to input/label count mismatch: Input {inputs.shape[0]}, Label {labels.shape[0]}")
                 continue # Skip problematic batch

            # Add check for expected input dimensions for model
            if inputs.shape[1] != model.chain_length or inputs.shape[2] != model.input_feature_dim:
                print(f"Skipping batch due to input dimension mismatch: Expected [B, {model.chain_length}, {model.input_feature_dim}], Got {inputs.shape}")
                continue


            optimizer.zero_grad()
            try:
                outputs = model(inputs) # [B, Num_Classes]
            except Exception as e:
                 print(f"\nError during model forward pass (training): {e}")
                 print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
                 continue # Skip batch if forward pass fails


            # Ensure output and label shapes match for loss calculation
            if outputs.shape[0] != labels.shape[0]:
                print(f"Skipping loss calculation due to output/label count mismatch: Output {outputs.shape[0]}, Label {labels.shape[0]}")
                continue

            if labels.max() >= model.num_classes or labels.min() < 0:
                 print(f"Skipping batch due to invalid label values: {labels.unique()}. Max class index is {model.num_classes-1}")
                 continue


            try:
                 loss = criterion(outputs, labels)
            except Exception as e:
                 print(f"\nError during loss calculation: {e}")
                 print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}, Labels: {labels}")
                 continue # Skip batch if loss calculation fails


            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar description
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.4f}' if train_total else 'N/A'
            })

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0
        train_pbar.close()


        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device).squeeze(1)

                if labels.ndim == 0: labels = labels.unsqueeze(0)
                if inputs.ndim == 2: inputs = inputs.unsqueeze(0)
                if inputs.shape[0] != labels.shape[0]: continue
                if inputs.shape[1] != model.chain_length or inputs.shape[2] != model.input_feature_dim: continue

                try:
                    outputs = model(inputs)
                except Exception as e:
                     print(f"\nError during model forward pass (validation): {e}")
                     print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
                     continue

                if outputs.shape[0] != labels.shape[0]: continue
                if labels.max() >= model.num_classes or labels.min() < 0: continue

                try:
                    loss = criterion(outputs, labels)
                except Exception as e:
                    print(f"\nError during loss calculation (validation): {e}")
                    print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}, Labels: {labels}")
                    continue

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({
                   'Loss': f'{loss.item():.4f}',
                   'Acc': f'{val_correct/val_total:.4f}' if val_total else 'N/A'
                })

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        val_pbar.close()

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f'Epoch {epoch+1}/{num_epochs} Results:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'  Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc:.4f}')

        # Check for improvement and save best model (based on validation loss)
        if epoch_val_loss < best_val_loss:
            print(f'  Validation loss improved ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model...')
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'  Validation loss did not improve. ({epochs_no_improve}/{patience})')

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
            break

    print("\nTraining finished.")
    # Load best model weights
    print(f"Loading best model from {model_save_path} (Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(torch.load(model_save_path))
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """Plots training and validation metrics."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    # plt.show() # Comment out if running non-interactively


def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate the model on test data."""
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    test_pbar = tqdm(test_loader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device).squeeze(1)

            if labels.ndim == 0: labels = labels.unsqueeze(0)
            if inputs.ndim == 2: inputs = inputs.unsqueeze(0)
            if inputs.shape[0] != labels.shape[0]: continue
            if inputs.shape[1] != model.chain_length or inputs.shape[2] != model.input_feature_dim: continue

            try:
                 outputs = model(inputs)
            except Exception as e:
                 print(f"\nError during model forward pass (evaluation): {e}")
                 print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")
                 continue # Skip batch if forward pass fails

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_pbar.close()

    if not all_labels:
        print("Warning: No valid predictions made during evaluation.")
        return 0.0, np.zeros((num_classes, num_classes), dtype=int)

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    return accuracy, cm


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Plots confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    # plt.show()


def predict_delay(model, flight_chain_data, preprocessor, device, class_names):
    """Predict delay level for a single flight chain."""
    if not preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted before prediction.")

    model = model.to(device)
    model.eval()

    # Preprocess flight chain
    X = preprocessor.transform_flight_chain(flight_chain_data) # Expects list of dicts/rows
    if X.shape[0] != model.chain_length:
        raise ValueError(f"Input flight chain length mismatch. Expected {model.chain_length}, got {X.shape[0]}")
    if X.shape[1] != preprocessor.total_dim:
         raise ValueError(f"Input feature dimension mismatch after preprocessing. Expected {preprocessor.total_dim}, got {X.shape[1]}")


    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)  # Add batch dimension [1, T, D]

    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    delay_level_index = predicted_idx.item()
    delay_level_name = class_names[delay_level_index]
    probs = probabilities.cpu().numpy()[0] # Probabilities for the single prediction

    print("\nPrediction Details:")
    print(f"  Predicted Delay Level: {delay_level_index} ({delay_level_name})")
    print("  Class Probabilities:")
    for i, name in enumerate(class_names):
        print(f"    {name}: {probs[i]:.4f}")

    return delay_level_index, delay_level_name, probs


# ================ MAIN EXECUTION ================

def main():
    # --- Configuration ---
    # DATA_DIR = 'processedDataTest/' # Original directory
    DATA_DIR = 'data/' # Example: If data is in a 'data' subdirectory
    PREPROCESSOR_PATH = 'fitted_preprocessor.joblib'
    MODEL_SAVE_PATH = 'best_simam_cnn_mogrifier_lstm.pth'

    # Ensure data directory exists
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and place CSV files inside.")
        print("Expected files: train_set.csv, validation_set.csv, test_set.csv")
        return # Exit if data dir is missing

    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_set.csv')
    VAL_DATA_PATH = os.path.join(DATA_DIR, 'validation_set.csv')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_set.csv')

    # Check if individual files exist
    for path in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]:
         if not os.path.isfile(path):
              print(f"Error: Required data file not found: {path}")
              return


    # Define features (Ensure these columns exist in your CSVs!)
    # Tail_Number, Schedule_DateTime, Flight_Duration_Minutes, Flight_Delay are used internally by Dataset
    categorical_features = ['Carrier_Airline', 'Origin', 'Dest', 'Orientation'] # Removed Tail_Number as it's mainly for grouping
    numerical_features = ['Flight_Duration_Minutes', 'FTD', 'PFD'] # Removed Flight_Delay (it's the target!)
    temporal_features = ['Schedule_DateTime']
    # Delay level definitions
    class_names = ['On Time/Early', 'Slight Delay', 'Moderate Delay', 'Significant Delay', 'Severe Delay']
    num_classes = len(class_names)

    # Model Hyperparameters
    CHAIN_LENGTH = 3       # T: Number of flights in a sequence
    LSTM_HIDDEN_SIZE = 128
    CNN_CHANNELS = [32, 64] # Output channels for CNN blocks
    # Use 2D Kernels: (time_kernel, feature_kernel)
    # Example: (1,3) = kernel across 1 time step, 3 features. (3,3) = 3 time steps, 3 features.
    CNN_KERNEL_SIZES = [(1, 3), (3, 3)] # Needs to be list of tuples/ints
    USE_SIMAM = True
    MOGRIFIER_ROUNDS = 5

    # Training Hyperparameters
    BATCH_SIZE = 32 # Reduced batch size for potentially large features
    NUM_EPOCHS = 10 # Reduced for faster testing, increase for real training (e.g., 30-50)
    LEARNING_RATE = 0.001
    PATIENCE = 5 # Early stopping patience

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")


    # --- Preprocessing ---
    # Try loading existing preprocessor, otherwise fit a new one
    if os.path.exists(PREPROCESSOR_PATH):
        print(f"Loading existing preprocessor from {PREPROCESSOR_PATH}...")
        preprocessor = FlightDataPreprocessor.load(PREPROCESSOR_PATH)
        # Verify loaded features match current config (optional but good practice)
        if (preprocessor.categorical_features != categorical_features or
            preprocessor.numerical_features != numerical_features or
            preprocessor.temporal_features != temporal_features):
             print("Warning: Loaded preprocessor features differ from config. Re-fitting...")
             preprocessor = FlightDataPreprocessor(categorical_features, numerical_features, temporal_features)
             train_df_for_fit = pd.read_csv(TRAIN_DATA_PATH) # Load fresh df
             preprocessor.fit(train_df_for_fit)
             preprocessor.save(PREPROCESSOR_PATH) # Save the newly fitted one

    else:
        print("No existing preprocessor found. Fitting a new one...")
        preprocessor = FlightDataPreprocessor(categorical_features, numerical_features, temporal_features)
        # Load only the training data for fitting
        train_df_for_fit = pd.read_csv(TRAIN_DATA_PATH)
        preprocessor.fit(train_df_for_fit)
        preprocessor.save(PREPROCESSOR_PATH) # Save the fitted preprocessor

    input_feature_dim = preprocessor.total_dim
    print(f"Input feature dimension for model: {input_feature_dim}")

    # --- Datasets and DataLoaders ---
    print("\nCreating datasets...")
    try:
        train_dataset = FlightChainDataset(TRAIN_DATA_PATH, preprocessor, CHAIN_LENGTH)
        val_dataset = FlightChainDataset(VAL_DATA_PATH, preprocessor, CHAIN_LENGTH)
        test_dataset = FlightChainDataset(TEST_DATA_PATH, preprocessor, CHAIN_LENGTH)
    except ValueError as e:
        print(f"Error creating datasets: {e}")
        return
    except FileNotFoundError as e:
         print(f"Error: {e}")
         return

    # Check if datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         print("Error: Training or Validation dataset is empty. Cannot proceed.")
         print("Please check your data files, feature definitions, and chain creation logic.")
         return

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Use num_workers based on available cores, handle potential issues on Windows/MacOS
    num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
    if os.name == 'nt': # Reduce workers on Windows due to potential overhead/issues
         num_workers = 0
    print(f"Using {num_workers} workers for DataLoaders.")

    # Consider persistent_workers=True if num_workers > 0 for faster epoch starts
    persist_workers = num_workers > 0

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True if DEVICE=='cuda' else False, persistent_workers=persist_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE=='cuda' else False, persistent_workers=persist_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True if DEVICE=='cuda' else False, persistent_workers=persist_workers)


    # --- Model Initialization ---
    print("\nInitializing model...")
    model = SimAMCNNMogrifierLSTM(
        input_feature_dim=input_feature_dim,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        num_classes=num_classes,
        cnn_channels=CNN_CHANNELS,
        kernel_sizes=CNN_KERNEL_SIZES,
        use_simam=USE_SIMAM,
        mogrifier_rounds=MOGRIFIER_ROUNDS,
        chain_length=CHAIN_LENGTH
    )
    # print(model) # Print model structure

    # --- Training ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        model_save_path=MODEL_SAVE_PATH,
        patience=PATIENCE
    )

    # --- Plotting Training History ---
    plot_training_history(history)

    # --- Evaluation ---
    print("\nEvaluating model on test set...")
    # Load the best model saved during training
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    test_acc, cm = evaluate_model(trained_model, test_loader, DEVICE, num_classes)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    # --- Plotting Confusion Matrix ---
    if test_acc > 0: # Only plot if evaluation produced results
        plot_confusion_matrix(cm, class_names)
    else:
        print("Skipping confusion matrix plot due to evaluation issues.")

    # --- Example Prediction ---
    print("\nExample Prediction:")
    if len(test_dataset) > 0:
        # Get a sample chain from the test set
        sample_idx = np.random.randint(0, len(test_dataset))
        sample_chain_tensor, sample_label_tensor = test_dataset[sample_idx]
        sample_chain_data = test_dataset.flight_chains[sample_idx] # Get original data (list of dicts)
        true_label_idx = sample_label_tensor.item()
        true_label_name = class_names[true_label_idx]

        print(f"Predicting for sample chain #{sample_idx} (True Label: {true_label_idx} - {true_label_name})")
        try:
            pred_idx, pred_name, probs = predict_delay(
                model=trained_model,
                flight_chain_data=sample_chain_data,
                preprocessor=preprocessor,
                device=DEVICE,
                class_names=class_names
            )
        except Exception as e:
             print(f"Error during example prediction: {e}")

    else:
        print("Test dataset is empty, cannot run example prediction.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
