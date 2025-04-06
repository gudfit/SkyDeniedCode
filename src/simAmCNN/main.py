import os
import sys
import pandas as pd
import numpy  as np
import torch
import pickle
import traceback
import json # Added for saving config

from torch.utils.data    import DataLoader
from tqdm                import tqdm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
# --- Configuration ---
OUTPUT_DIR             = os.path.normpath(os.path.join(script_dir, '../../mlData/simResults/'))
ADAPTED_DATA_DIR       = os.path.normpath(os.path.join(script_dir, '../../mlData/adaptedData/'))
MODELS_DIR             = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR              = os.path.join(OUTPUT_DIR, 'plots')
PREPROCESSOR_SAVE_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')
MODEL_SAVE_PATH        = os.path.join(MODELS_DIR, 'best_model.pth')
CONFIG_SAVE_PATH       = os.path.join(MODELS_DIR, 'model_config.json')
HISTORY_PLOT_PATH      = os.path.join(PLOTS_DIR, 'training_history.png')
CONFUSION_MATRIX_PATH  = os.path.join(PLOTS_DIR, 'confusion_matrix.png')

CATEGORICAL_FEATURES   = ['Carrier_Airline', 'Origin', 'Dest', 'Orientation', 'Tail_Number'] 
NUMERICAL_FEATURES     = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Flight_Delay'] 
TEMPORAL_FEATURES      = ['Schedule_DateTime']

# Model Hyperparameters
LSTM_HIDDEN_SIZE       = 128
CNN_CHANNELS           = [32, 64]
CNN_KERNEL_SIZES       = [(1, 1), (3, 3)] # List of tuples for 2D kernels
USE_SIMAM              = True
LAMBDA_VAL             = 1e-4 # Adjusted SimAM lambda based on common practice
MOGRIFIER_ROUNDS       = 5 # Adjusted based on common practice

# Training Hyperparameters
BATCH_SIZE             = 16 # Keep small for testing/debugging memory
NUM_EPOCHS             = 5  # Keep low for testing
LEARNING_RATE          = 0.001
PATIENCE               = 3 # Early stopping patience (adjust as needed)


# Custom Dataset
class SimpleFlightChainDataset(torch.utils.data.Dataset):
    def __init__(self, chains, labels, preprocessor, input_dim, chain_length):
        self.chains                 = chains
        self.labels                 = labels
        self.preprocessor           = preprocessor
        self.input_dim              = input_dim
        self.chain_length           = chain_length # Store expected chain length

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        chain                       = self.chains[idx]
        label                       = self.labels[idx]

        current_chain_len           = len(chain)
        if current_chain_len       != self.chain_length:
            print(f"Warning: Chain {idx} has length {current_chain_len}, expected {self.chain_length}. Skipping or Padding needed.")
            pass


        # Create a zero tensor of the right size [T, D]
        X = np.zeros((self.chain_length, self.input_dim), dtype=np.float32)

        # Process each flight in the chain
        for i, flight in enumerate(chain):
             # Only process up to the expected chain length
            if i >= self.chain_length:
                 break
            try:
                # Convert to dict if it's a Series or other object
                flight_data         = flight if isinstance(flight, dict) else flight.to_dict()
                # Get features for this flight
                features            = self.preprocessor.transform_single_flight(flight_data)
                # Ensure feature vector dimension matches input_dim
                feat_len            = len(features)
                if feat_len        != self.input_dim:
                     # This signals an issue in the preprocessor's transform consistency
                     print(f"Warning: Feature vector dim mismatch for flight {i} in chain {idx}. Expected {self.input_dim}, got {feat_len}. Padding/Truncating.")
                     min_dim        = min(feat_len, self.input_dim)
                     X[i, :min_dim] = features[:min_dim]
                else:
                     X[i, :]        = features

            except Exception as e:
                print(f"Error processing flight {i} in chain {idx}: {e}")
                # Fill row with zeros in case of error
                X[i, :]             = np.zeros(self.input_dim, dtype=np.float32)

        # Ensure label is within expected range
        if not (0 <= label < 5):
             print(f"Warning: Invalid label {label} at index {idx}. Setting to 0.")
             label = 0


        return torch.FloatTensor(X), torch.LongTensor([label])

def main():
    """Main function to train the flight delay prediction model using adapted data"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # Create output directories if they don't exist
    for directory in [MODELS_DIR, PLOTS_DIR, ADAPTED_DATA_DIR]:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

    try:
        # --- Load Adapted Data ---
        print("Loading adapted data...")
        train_data_path                = os.path.join(ADAPTED_DATA_DIR, 'train_data.pkl')
        val_data_path                  = os.path.join(ADAPTED_DATA_DIR, 'val_data.pkl')
        test_data_path                 = os.path.join(ADAPTED_DATA_DIR, 'test_data.pkl')

        if not all(os.path.exists(p) for p in [train_data_path, val_data_path, test_data_path]):
             print(f"Error: Pickle files not found in {ADAPTED_DATA_DIR}. Please run data preparation first.")
             return

        with open(train_data_path, 'rb') as f:
            train_chains, train_labels = pickle.load(f)

        with open(val_data_path, 'rb') as f:
            val_chains, val_labels     = pickle.load(f)

        with open(test_data_path, 'rb') as f:
            test_chains, test_labels   = pickle.load(f)

        print(f"Loaded {len(train_chains)} training chains, {len(val_chains)} validation chains, and {len(test_chains)} test chains")

        # --- Determine Chain Length (T) ---
        if not train_chains:
             print("Error: Training data is empty!")
             return
        # Assuming all chains have the same length, take from the first one
        chain_length = len(train_chains[0])
        print(f"Determined chain length (T): {chain_length}")


        if 'src' not in sys.path:
             sys.path.insert(0, 'src')

        try:
            from flight_delay_model import (
                FlightDataPreprocessor,
                SimAMCNNMogrifierLSTM,
                train_model,
                evaluate_model,
                plot_training_history,
                plot_confusion_matrix
            )
            print("Successfully imported model components from src/flight_delay_model.py")
        except ImportError as e:
            print(f"Error importing from src/flight_delay_model.py: {e}")
            print("Please ensure the file exists and contains the required classes/functions.")
            return
        except Exception as e:
             print(f"An unexpected error occurred during import: {e}")
             traceback.print_exc()
             return


        # --- Initialize and Fit Preprocessor ---
        print("Initializing preprocessor...")
        preprocessor = FlightDataPreprocessor(
            categorical_features = CATEGORICAL_FEATURES,
            numerical_features   = NUMERICAL_FEATURES, 
            temporal_features    = TEMPORAL_FEATURES
        )

        print("Fitting preprocessor on training data...")
        # Flatten the chains for preprocessing fitting
        all_training_flights     = []
        for chain in train_chains:
            for flight in chain:
                 # Convert to dict
                 flight_dict     = flight if isinstance(flight, dict) else flight.to_dict()
                 all_training_flights.append(flight_dict)

        if not all_training_flights:
             print("Error: No flights found in training chains to fit preprocessor.")
             return

        train_df_for_fitting     = pd.DataFrame(all_training_flights)

        # Ensure all expected columns exist, fill if necessary before fitting
        print("Checking/preparing DataFrame columns for preprocessor fitting...")
        for feature in CATEGORICAL_FEATURES:
            if feature not in train_df_for_fitting.columns:
                print(f"Warning: Categorical feature '{feature}' not found in fitting data. Adding with 'Unknown'.")
                train_df_for_fitting[feature]           = 'Unknown'
            else:
                 # Fill NaNs in existing categorical columns before fitting OHE
                 train_df_for_fitting[feature]          = train_df_for_fitting[feature].fillna('Missing')


        for feature in NUMERICAL_FEATURES:
            if feature not in train_df_for_fitting.columns:
                print(f"Warning: Numerical feature '{feature}' not found in fitting data. Adding with 0.0.")
                train_df_for_fitting[feature]           = 0.0
            else:
                # Fill NaNs in existing numerical columns with mean before fitting Scaler
                mean_val = train_df_for_fitting[feature].mean()
                train_df_for_fitting[feature]           = train_df_for_fitting[feature].fillna(mean_val)


        if TEMPORAL_FEATURES[0] not in train_df_for_fitting.columns: 
            print(f"Warning: Temporal feature '{TEMPORAL_FEATURES[0]}' not found. Adding default.")
            train_df_for_fitting[TEMPORAL_FEATURES[0]]  = pd.Timestamp('2000-01-01')
        else:
            # Convert to datetime and fill NaNs before fitting temporal processing
             train_df_for_fitting[TEMPORAL_FEATURES[0]] = pd.to_datetime(train_df_for_fitting[TEMPORAL_FEATURES[0]], errors='coerce')
             train_df_for_fitting[TEMPORAL_FEATURES[0]] = train_df_for_fitting[TEMPORAL_FEATURES[0]].fillna(pd.Timestamp('2000-01-01'))


        # Fit the preprocessor
        preprocessor.fit(train_df_for_fitting)
        actual_input_dim         = preprocessor.total_dim # Get dimension directly from preprocessor
        print(f"Preprocessor fitted. Determined Input Feature Dimension (D): {actual_input_dim}")


        # --- Create Datasets ---
        print(f"Creating datasets using input dimension: {actual_input_dim}")
        train_dataset            = SimpleFlightChainDataset(train_chains, train_labels, preprocessor, actual_input_dim, chain_length)
        val_dataset              = SimpleFlightChainDataset(val_chains,   val_labels,   preprocessor, actual_input_dim, chain_length)
        test_dataset             = SimpleFlightChainDataset(test_chains,  test_labels,  preprocessor, actual_input_dim, chain_length)

        # --- Create DataLoaders ---
        # Handle num_workers based on OS
        num_workers              = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
        if os.name              == 'nt': num_workers = 0
        print(f"Using batch size: {BATCH_SIZE}, num_workers: {num_workers}")

        train_loader             = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers)
        val_loader               = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
        test_loader              = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


        # --- Initialize Model ---
        num_classes              = 5
        print(f"Initializing model with Input Dim (D): {actual_input_dim}, Chain Length (T): {chain_length}")

        model                    = SimAMCNNMogrifierLSTM(
            input_feature_dim    = actual_input_dim,   # D
            lstm_hidden_size     = LSTM_HIDDEN_SIZE,
            num_classes          = num_classes,
            cnn_channels         = CNN_CHANNELS,
            kernel_sizes         = CNN_KERNEL_SIZES,   # Pass list of tuples for 2D kernels
            use_simam            = USE_SIMAM,
            lambda_val           = LAMBDA_VAL,         # Use adjusted lambda
            mogrifier_rounds     = MOGRIFIER_ROUNDS,   # Use adjusted rounds
            chain_length         = chain_length        # T
        )

        # --- Training Setup ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model.to(device) 

        criterion                = torch.nn.CrossEntropyLoss()
        optimizer                = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # --- Train Model ---
        print(f"Starting training for {NUM_EPOCHS} epochs...")
        trained_model, history   = train_model(
            model                = model,
            train_loader         = train_loader,
            val_loader           = val_loader,
            criterion            = criterion,
            optimizer            = optimizer,
            num_epochs           = NUM_EPOCHS,
            model_save_path      = MODEL_SAVE_PATH, # Use defined path
            device               = device,
            patience             = PATIENCE         # Pass patience for early stopping
        )

        # --- Plot Training History ---
        print(f"Plotting training history to {HISTORY_PLOT_PATH}...")
        plot_training_history(history, save_path=HISTORY_PLOT_PATH) # Use defined path

        # --- Evaluation ---
        print("Evaluating model on test set (using best model saved during training)...")
        # Load the best model explicitly
        try:
             model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
             model.to(device)
             test_acc, cm        = evaluate_model(model=model, test_loader=test_loader, device=device, num_classes=num_classes) # Pass num_classes
             print(f'Test Accuracy: {test_acc:.4f}')

             # --- Plot Confusion Matrix ---
             class_names         = ['On Time/Early', 'Slight Delay', 'Moderate Delay', 'Significant Delay', 'Severe Delay']
             if test_acc         > 0 or np.sum(cm) > 0: # Plot if there are results
                  print(f"Plotting confusion matrix to {CONFUSION_MATRIX_PATH}...")
                  plot_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PATH)
             else:
                  print("Skipping confusion matrix plot due to no evaluation results.")

        except FileNotFoundError:
             print(f"Error: Best model file not found at {MODEL_SAVE_PATH}. Cannot evaluate.")
        except Exception as e:
             print(f"An error occurred during evaluation or plotting: {e}")
             traceback.print_exc()


        # --- Save Artifacts ---
        # Save preprocessor (fitted)
        try:
            print(f"Saving fitted preprocessor to {PREPROCESSOR_SAVE_PATH}...")
            preprocessor.save(PREPROCESSOR_SAVE_PATH)
        except AttributeError:
             print("Warning: Preprocessor does not have a 'save' method. Saving with pickle.")
             with open(PREPROCESSOR_SAVE_PATH, 'wb') as f:
                 pickle.dump(preprocessor, f)
        except Exception as e:
             print(f"Error saving preprocessor: {e}")


        # Save model config
        print(f"Saving model configuration to {CONFIG_SAVE_PATH}...")
        model_config = {
            'input_feature_dim' : actual_input_dim,
            'lstm_hidden_size'  : LSTM_HIDDEN_SIZE,
            'num_classes'       : num_classes,
            'cnn_channels'      : CNN_CHANNELS,
            'kernel_sizes'      : [list(ks) for ks in CNN_KERNEL_SIZES], # Convert tuples to lists for JSON
            'use_simam'         : USE_SIMAM,
            'lambda_val'        : LAMBDA_VAL,
            'mogrifier_rounds'  : MOGRIFIER_ROUNDS,
            'chain_length'      : chain_length
        }
        try:
            with open(CONFIG_SAVE_PATH, 'w') as f:
                json.dump(model_config, f, indent=4)
        except Exception as e:
             print(f"Error saving model config: {e}")


        print(f"\nDone! Artifacts saved to {MODELS_DIR} and {PLOTS_DIR}")

    except FileNotFoundError as e:
         print(f"Error: Required file not found. {e}")
         traceback.print_exc()
    except ImportError       as e:
         print(f"Error: Failed to import necessary modules. {e}")
         traceback.print_exc()
    except ValueError        as e:
         print(f"Error: A value error occurred (check data shapes/types). {e}")
         traceback.print_exc()
    except RuntimeError      as e:
         print(f"Error: A runtime error occurred (often CUDA/GPU related). {e}")
         traceback.print_exc()
    except Exception         as e:
        print(f"An unexpected error occurred in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
