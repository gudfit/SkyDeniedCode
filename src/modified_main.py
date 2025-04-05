import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import traceback

def main():
    """Main function to train the flight delay prediction model using adapted data."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories if they don't exist
    for directory in ['models', 'plots', 'adapted_data']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    try:
        # Load the adapted data
        print("Loading adapted data...")
        with open('adapted_data/train_data.pkl', 'rb') as f:
            train_chains, train_labels = pickle.load(f)
        
        with open('adapted_data/val_data.pkl', 'rb') as f:
            val_chains, val_labels = pickle.load(f)
        
        with open('adapted_data/test_data.pkl', 'rb') as f:
            test_chains, test_labels = pickle.load(f)
        
        print(f"Loaded {len(train_chains)} training chains, {len(val_chains)} validation chains, and {len(test_chains)} test chains")
        
        # Import the model after fixing the flight_delay_model.py file
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
        except ImportError as e:
            print(f"Error importing model: {e}")
            return
        
        # Define feature sets
        categorical_features = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation']
        numerical_features = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Flight_Delay']
        temporal_features = ['Schedule_DateTime']
        
        # Define embedding dimensions for categorical features with many levels
        embedding_dims = {
            'Tail_Number': 16,
            'Origin': 8,
            'Dest': 8
        }
        
        # Initialize preprocessor
        preprocessor = FlightDataPreprocessor(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            temporal_features=temporal_features,
            embedding_dims=embedding_dims
        )
        
        # Fit preprocessor on training data
        print("Fitting preprocessor...")
        # Flatten the chains for preprocessing
        all_flights = []
        for chain in train_chains:
            all_flights.extend(chain)
        
        # Create a dataframe from all flights for preprocessing
        flattened_data = []
        for flight in all_flights:
            # Convert to dict if it's not already
            flight_dict = flight if isinstance(flight, dict) else flight.to_dict()
            flattened_data.append(flight_dict)
        
        train_df = pd.DataFrame(flattened_data)
        
        # Fill in missing values with defaults to prevent preprocessing errors
        for feature in categorical_features:
            if feature not in train_df.columns:
                train_df[feature] = 'Unknown'
        
        for feature in numerical_features:
            if feature not in train_df.columns:
                train_df[feature] = 0.0
        
        # Ensure Schedule_DateTime is present and in correct format
        if 'Schedule_DateTime' not in train_df.columns:
            train_df['Schedule_DateTime'] = pd.Timestamp('2022-01-01')
        else:
            train_df['Schedule_DateTime'] = pd.to_datetime(train_df['Schedule_DateTime'])
        
        # Fit the preprocessor
        preprocessor.fit(train_df)
        
        # *** IMPORTANT: Get actual feature vector dimension ***
        # Process a single flight to get the actual dimension
        test_vector = preprocessor.transform_single_flight(train_df.iloc[0])
        actual_dim = len(test_vector)
        print(f"Actual feature vector dimension: {actual_dim}")
        
        # Use a simple dataset class with the ACTUAL dimension
        class SimpleFlightChainDataset(torch.utils.data.Dataset):
            def __init__(self, chains, labels, preprocessor, input_dim):
                self.chains = chains
                self.labels = labels
                self.preprocessor = preprocessor
                self.input_dim = input_dim
            
            def __len__(self):
                return len(self.chains)
            
            def __getitem__(self, idx):
                chain = self.chains[idx]
                label = self.labels[idx]
                
                # Create a zero tensor of the right size
                X = np.zeros((len(chain), self.input_dim))
                
                # Process each flight in the chain
                for i, flight in enumerate(chain):
                    try:
                        # Get features for this flight
                        features = self.preprocessor.transform_single_flight(flight)
                        
                        # Ensure it fits in our tensor by truncating or padding
                        min_dim = min(len(features), self.input_dim)
                        X[i, :min_dim] = features[:min_dim]
                    except Exception as e:
                        print(f"Error processing flight {i} in chain {idx}: {e}")
                
                return torch.FloatTensor(X), torch.LongTensor([label])
        
        # Create datasets using the ACTUAL dimension we measured
        print(f"Creating datasets with actual dimension: {actual_dim}")
        train_dataset = SimpleFlightChainDataset(train_chains, train_labels, preprocessor, actual_dim)
        val_dataset = SimpleFlightChainDataset(val_chains, val_labels, preprocessor, actual_dim)
        test_dataset = SimpleFlightChainDataset(test_chains, test_labels, preprocessor, actual_dim)
        
        # Create data loaders with small batch size for testing
        batch_size = 16
        print(f"Using batch size: {batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model with the ACTUAL input size
        input_size = actual_dim
        hidden_size = 128
        num_classes = 5  # Delay levels (0-4)
        
        print(f"Initializing model with input size: {input_size}")
        
        model = SimAMCNNMogrifierLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            cnn_channels=[32, 64],
            kernel_sizes=[1, 3],
            use_simam=True,
            lambda_val=0.1,
            num_rounds=6
        )
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Train model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Set num_epochs very low for testing
        num_epochs = 5
        print(f"Training for {num_epochs} epochs")
        
        # Train the model
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            model_save_path='models/best_model.pth',
            device=device
        )
        
        # Plot training history
        plot_training_history(history)
        plt.savefig('plots/training_history.png')
        
        # Load the best model for evaluation
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        test_acc, cm = evaluate_model(model, test_loader, device=device)
        print(f'Test Accuracy: {test_acc:.4f}')
        
        # Plot confusion matrix
        class_names = ['On Time', 'Slight Delay', 'Minor Delay', 'Moderate Delay', 'Severe Delay']
        plot_confusion_matrix(cm, class_names)
        plt.savefig('plots/confusion_matrix.png')
        
        # Save preprocessor for later use
        with open('models/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        
        # Save model config
        import json
        model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'cnn_channels': [32, 64],
            'kernel_sizes': [1, 3],
            'use_simam': True,
            'lambda_val': 0.1,
            'num_rounds': 6
        }
        
        with open('models/model_config.json', 'w') as f:
            json.dump(model_config, f, indent=4)
        
        print("Done! Model, preprocessor, and config saved to models/ directory.")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
