import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the model components
from flight_delay_model import (
    FlightDataPreprocessor,
    FlightChainDataset,
    SimAMCNNMogrifierLSTM,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix
)

def create_flight_chains(df, min_chain_length=3, max_time_diff_hours=24):
    """
    Create chains of consecutive flights for the same aircraft.
    
    Args:
        df: DataFrame containing flight data
        min_chain_length: Minimum number of flights in a chain
        max_time_diff_hours: Maximum allowed time difference between consecutive flights
        
    Returns:
        List of flight chains (each chain is a list of flight data dicts)
    """
    print("Creating flight chains...")
    flight_chains = []
    
    # Group flights by tail number and sort by schedule time
    for tail_number, group in tqdm(df.groupby('Tail_Number')):
        sorted_flights = group.sort_values('Schedule_DateTime')
        
        # Skip if not enough flights for this tail number
        if len(sorted_flights) < min_chain_length:
            continue
        
        # Initialize the current chain
        current_chain = [sorted_flights.iloc[0].to_dict()]
        
        # Process remaining flights
        for i in range(1, len(sorted_flights)):
            prev_flight = sorted_flights.iloc[i-1]
            curr_flight = sorted_flights.iloc[i]
            
            # Calculate time difference
            prev_time = pd.to_datetime(prev_flight['Schedule_DateTime']) + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])
            curr_time = pd.to_datetime(curr_flight['Schedule_DateTime'])
            time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
            
            # Check if this flight should be added to the current chain
            if time_diff <= max_time_diff_hours:
                current_chain.append(curr_flight.to_dict())
            else:
                # If current chain is long enough, add it to flight_chains
                if len(current_chain) >= min_chain_length:
                    flight_chains.append(current_chain)
                
                # Start a new chain
                current_chain = [curr_flight.to_dict()]
        
        # Add the last chain if it's long enough
        if len(current_chain) >= min_chain_length:
            flight_chains.append(current_chain)
    
    print(f"Created {len(flight_chains)} flight chains")
    return flight_chains

def prepare_data_for_model(flight_chains, chain_length=3):
    """
    Prepare data for model training by creating shorter chains of the specified length.
    
    Args:
        flight_chains: List of flight chains
        chain_length: Target chain length for model training
        
    Returns:
        List of chains with the specified length and their corresponding labels
    """
    model_chains = []
    delay_labels = []
    
    for chain in flight_chains:
        # Skip if chain is shorter than required
        if len(chain) < chain_length:
            continue
        
        # Create chains of the specified length by sliding a window
        for i in range(len(chain) - chain_length + 1):
            sub_chain = chain[i:i+chain_length]
            
            # Extract delay label for the last flight
            last_flight = sub_chain[-1]
            delay_minutes = last_flight['Flight_Delay']
            
            # Convert delay to label
            if delay_minutes <= 0:
                label = 0  # On time or early
            elif delay_minutes <= 15:
                label = 1  # Slight delay
            elif delay_minutes <= 30:
                label = 2  # Minor delay
            elif delay_minutes <= 60:
                label = 3  # Moderate delay
            else:
                label = 4  # Severe delay
            
            model_chains.append(sub_chain)
            delay_labels.append(label)
    
    return model_chains, delay_labels

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories if they don't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv('processedDataTest/processed_flights_full_data.csv')
    
    # Ensure required columns exist
    required_columns = [
        'Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Schedule_DateTime',
        'Flight_Duration_Minutes', 'Flight_Delay', 'Orientation', 'FTD', 'PFD'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in the dataset: {missing_columns}")
        # If columns are missing, try to create them or adapt
        for col in missing_columns:
            if col == 'FTD':  # Flight Time Difference - estimate if missing
                df['FTD'] = 0  # Default value
            elif col == 'PFD':  # Previous Flight Delay - estimate if missing
                df['PFD'] = 0  # Default value
            elif col == 'Orientation':  # Flight direction - estimate if missing
                df['Orientation'] = 'Unknown'  # Default value
    
    # Convert date columns to datetime if they are strings
    if isinstance(df['Schedule_DateTime'].iloc[0], str):
        df['Schedule_DateTime'] = pd.to_datetime(df['Schedule_DateTime'])
    
    # Create flight chains
    flight_chains = create_flight_chains(df)
    
    # Prepare data for model
    print("Preparing data for model...")
    model_chains, delay_labels = prepare_data_for_model(flight_chains)
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        model_chains, delay_labels, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 of 0.8 = 0.2 of total
    )
    
    print(f"Train set: {len(X_train)} chains")
    print(f"Validation set: {len(X_val)} chains")
    print(f"Test set: {len(X_test)} chains")
    
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
    # Create a dataframe from all flights in training chains
    all_train_flights = []
    for chain in X_train:
        all_train_flights.extend(chain)
    train_df = pd.DataFrame(all_train_flights)
    preprocessor.fit(train_df)
    
    # Create custom datasets
    class CustomFlightChainDataset(torch.utils.data.Dataset):
        def __init__(self, chains, labels, preprocessor):
            self.chains = chains
            self.labels = labels
            self.preprocessor = preprocessor
        
        def __len__(self):
            return len(self.chains)
        
        def __getitem__(self, idx):
            chain = self.chains[idx]
            label = self.labels[idx]
            
            # Transform the flight chain to matrix X
            X = self.preprocessor.transform_flight_chain(chain)
            
            return torch.FloatTensor(X), torch.LongTensor([label])
    
    # Create datasets
    train_dataset = CustomFlightChainDataset(X_train, y_train, preprocessor)
    val_dataset = CustomFlightChainDataset(X_val, y_val, preprocessor)
    test_dataset = CustomFlightChainDataset(X_test, y_test, preprocessor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    input_size = preprocessor.total_dim
    hidden_size = 128
    num_classes = 5  # Delay levels (0-4)
    
    print(f"Input size: {input_size}")
    
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
    
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=30,
        model_save_path='models/best_model.pth',
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    plt.savefig('plots/training_history.png')
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('models/best_model.pth'))
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_acc, cm = evaluate_model(model, test_loader, device=device)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Plot confusion matrix
    class_names = ['On Time', 'Slight Delay', 'Minor Delay', 'Moderate Delay', 'Severe Delay']
    plot_confusion_matrix(cm, class_names)
    plt.savefig('plots/confusion_matrix.png')
    
    print("Done!")

if __name__ == "__main__":
    main()
