import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
adapted_data_dir = os.path.normpath(os.path.join(script_dir, '../../mlData/adaptedData'))
class FlightDataAdapter:
    """
    Adapter to transform processed flight chain data into the format needed by the model.
    """
    def __init__(self, categorical_features, numerical_features, temporal_features):
        """
        Initialize the adapter with feature definitions.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            temporal_features: List of temporal features
        """
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.temporal_features = temporal_features
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
    
    def fit(self, df):
        """
        Fit the encoders and scalers on the training data.
        
        Args:
            df: DataFrame containing the training data
        """
        # Extract flight-specific features for all flights in chains
        flight_dfs = []
        
        for flight_idx in range(1, 4):  # Assuming 3 flights in a chain
            prefix = f"flight{flight_idx}_"
            
            # Get all columns for this flight
            flight_cols = [col for col in df.columns if col.startswith(prefix)]
            
            # Create a DataFrame with just this flight's data
            flight_df = df[flight_cols].copy()
            
            # Rename columns to remove the prefix
            flight_df.columns = [col.replace(prefix, "") for col in flight_df.columns]
            
            flight_dfs.append(flight_df)
        
        # Combine all flights into a single DataFrame
        all_flights_df = pd.concat(flight_dfs, ignore_index=True)
        
        # Fit categorical encoders
        for feature in self.categorical_features:
            if feature in all_flights_df.columns:
                encoder = OneHotEncoder(handle_unknown='ignore')
                encoder.fit(all_flights_df[feature].values.reshape(-1, 1))
                self.categorical_encoders[feature] = encoder
        
        # Fit numerical scaler
        if all(feat in all_flights_df.columns for feat in self.numerical_features):
            self.numerical_scaler.fit(all_flights_df[self.numerical_features])
        
        return self
    
    def transform_chain(self, chain_row):
        """
        Transform a single row of flight chain data into a format suitable for the model.
        
        Args:
            chain_row: A row from the flight chain DataFrame
            
        Returns:
            List of dictionaries, each containing a flight's features
        """
        flights = []
        
        for flight_idx in range(1, 4):  # Assuming 3 flights in a chain
            prefix = f"flight{flight_idx}_"
            
            # Get all columns for this flight
            flight_cols = [col for col in chain_row.index if col.startswith(prefix)]
            
            # Create a dictionary with this flight's data
            flight_data = {}
            for col in flight_cols:
                # Remove the prefix from the column name
                feature_name = col.replace(prefix, "")
                flight_data[feature_name] = chain_row[col]
            
            # Add FTD and PFD if they exist separately in the chain data
            if f"flight{flight_idx}_FTD" not in chain_row.index and f"flight{flight_idx}_PFD" not in chain_row.index:
                if flight_idx > 1:
                    if f"flight{flight_idx}_PFD" not in flight_data and "PFD" not in flight_data:
                        flight_data["PFD"] = chain_row.get(f"flight{flight_idx}_PFD", 
                                              chain_row.get(f"flight{flight_idx-1}_Flight_Delay", 0))
                    
                    if f"flight{flight_idx}_FTD" not in flight_data and "FTD" not in flight_data:
                        flight_data["FTD"] = chain_row.get(f"ground_time_{flight_idx-1}", 0)
            
            flights.append(flight_data)
        
        return flights
    
    def process_data(self, df):
        """
        Process the entire flight chain dataset.
        
        Args:
            df: DataFrame containing the flight chain data
            
        Returns:
            List of flight chains and list of delay labels
        """
        chains = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing flight chains"):
            # Transform the chain
            flight_chain = self.transform_chain(row)
            
            # Get the delay label
            if "delay_category" in row:
                label = row["delay_category"]
            elif "delay_label" in row:
                # Convert delay to categorical if needed
                delay = row["delay_label"]
                if delay <= 0:
                    label = 0  # On time or early
                elif delay <= 15:
                    label = 1  # Slight delay
                elif delay <= 30:
                    label = 2  # Minor delay
                elif delay <= 60:
                    label = 3  # Moderate delay
                else:
                    label = 4  # Severe delay
            else:
                # Assume last flight's delay as label if no explicit label
                last_flight = flight_chain[-1]
                delay = last_flight.get("Flight_Delay", 0)
                if delay <= 0:
                    label = 0
                elif delay <= 15:
                    label = 1
                elif delay <= 30:
                    label = 2
                elif delay <= 60:
                    label = 3
                else:
                    label = 4
            
            chains.append(flight_chain)
            labels.append(label)
        
        return chains, labels

class AdaptedFlightChainDataset(Dataset):
    """
    Dataset adapter for flight chains.
    """
    def __init__(self, chains, labels, preprocessor):
        """
        Args:
            chains: List of flight chains
            labels: List of delay labels
            preprocessor: FlightDataPreprocessor instance
        """
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

def main():
    # Define feature sets
    categorical_features = ['Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 'Orientation']
    numerical_features = ['Flight_Duration_Minutes', 'FTD', 'PFD', 'Flight_Delay']
    temporal_features = ['Schedule_DateTime']
    # Load the processed data
    train_df = pd.read_csv(os.path.normpath(os.path.join(script_dir, '../../mlData/processedDataTest/train_set.csv')))
    val_df = pd.read_csv(os.path.normpath(os.path.join(script_dir, '../../mlData/processedDataTest/validation_set.csv')))
    test_df = pd.read_csv(os.path.normpath(os.path.join(script_dir, '../../mlData/processedDataTest/test_set.csv')))
   
    # Create data adapter
    adapter = FlightDataAdapter(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        temporal_features=temporal_features
    )
    
    # Fit the adapter on training data
    adapter.fit(train_df)
    
    # Process the data
    print("Processing training data...")
    train_chains, train_labels = adapter.process_data(train_df)
    
    print("Processing validation data...")
    val_chains, val_labels = adapter.process_data(val_df)
    
    print("Processing test data...")
    test_chains, test_labels = adapter.process_data(test_df)
    
    print(f"Processed {len(train_chains)} training chains, {len(val_chains)} validation chains, and {len(test_chains)} test chains")
    
    # Create adapter output directory
    if not os.path.exists(adapted_data_dir):
        os.makedirs(adapted_data_dir)
    
    train_data_path = os.path.join(adapted_data_dir, 'train_data.pkl')
    with open(train_data_path, 'wb') as f:
        pickle.dump((train_chains, train_labels), f)
    # Save validation data
    val_data_path = os.path.join(adapted_data_dir, 'val_data.pkl')
    with open(val_data_path, 'wb') as f:
        pickle.dump((val_chains, val_labels), f)

    # Save test data
    test_data_path = os.path.join(adapted_data_dir, 'test_data.pkl')
    with open(test_data_path, 'wb') as f:
        pickle.dump((test_chains, test_labels), f)    
    
    # Sample check: Show the structure of the first chain
    print("\nSample of first chain in training data:")
    print(f"Number of flights in chain: {len(train_chains[0])}")
    for i, flight in enumerate(train_chains[0]):
        print(f"Flight {i+1} features: {list(flight.keys())[:5]}...")
    
    print(f"Label for this chain: {train_labels[0]}")
    
    print("\nData adaptation completed successfully!")

if __name__ == "__main__":
    main()
