import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

def clean_and_process_flight_data(input_file, output_file):
    """
    Clean and process raw flight data.
    
    Args:
        input_file: Path to the raw flight data CSV
        output_file: Path to save the processed data
    """
    print(f"Processing {input_file}...")
    
    # Read the data
    df = pd.read_csv(input_file, low_memory=False)
    
    # Select relevant columns
    relevant_columns = [
        'Year', 'Month', 'DayofMonth', 'DayOfWeek', 
        'FlightDate', 'Reporting_Airline', 'Tail_Number',
        'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay',
        'CRSArrTime', 'ArrTime', 'ArrDelay', 'Cancelled', 'Diverted',
        'CRSElapsedTime', 'ActualElapsedTime', 'Distance'
    ]
    
    # Check which columns exist in the dataset
    available_columns = [col for col in relevant_columns if col in df.columns]
    missing_columns = set(relevant_columns) - set(available_columns)
    
    if missing_columns:
        print(f"Warning: The following columns are missing: {missing_columns}")
    
    # Filter to available columns
    df = df[available_columns].copy()
    
    # Remove cancelled or diverted flights
    if 'Cancelled' in df.columns:
        df = df[df['Cancelled'] == 0]
    if 'Diverted' in df.columns:
        df = df[df['Diverted'] == 0]
    
    # Filter out rows with missing values in critical columns
    critical_columns = ['Tail_Number', 'Origin', 'Dest']
    critical_columns = [col for col in critical_columns if col in df.columns]
    df = df.dropna(subset=critical_columns)
    
    # Clean tail numbers
    if 'Tail_Number' in df.columns:
        # Remove rows with invalid tail numbers
        df = df[df['Tail_Number'].notna() & (df['Tail_Number'] != '')]
        # Remove any whitespace
        df['Tail_Number'] = df['Tail_Number'].str.strip()
    
    # Create datetime columns
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])
    
    # Convert time format (HHMM) to proper time
    if 'CRSDepTime' in df.columns:
        df['CRSDepTime'] = df['CRSDepTime'].astype(str).str.zfill(4)
        df['CRSDepTimeHour'] = df['CRSDepTime'].str[:2].astype(int)
        df['CRSDepTimeMinute'] = df['CRSDepTime'].str[2:].astype(int)
    
    if 'CRSArrTime' in df.columns:
        df['CRSArrTime'] = df['CRSArrTime'].astype(str).str.zfill(4)
        df['CRSArrTimeHour'] = df['CRSArrTime'].str[:2].astype(int)
        df['CRSArrTimeMinute'] = df['CRSArrTime'].str[2:].astype(int)
    
    # Create proper datetime for scheduled departures and arrivals
    df['Schedule_DateTime'] = df.apply(
        lambda row: pd.Timestamp(
            year=row['FlightDate'].year,
            month=row['FlightDate'].month,
            day=row['FlightDate'].day,
            hour=row['CRSDepTimeHour'],
            minute=row['CRSDepTimeMinute']
        ),
        axis=1
    )
    
    # Calculate flight duration in minutes
    if 'CRSElapsedTime' in df.columns:
        df['Flight_Duration_Minutes'] = df['CRSElapsedTime']
    else:
        # Estimate flight duration based on scheduled departure and arrival times
        df['Flight_Duration_Minutes'] = ((df['CRSArrTimeHour'] * 60 + df['CRSArrTimeMinute']) - 
                                         (df['CRSDepTimeHour'] * 60 + df['CRSDepTimeMinute']))
        # Handle overnight flights
        df.loc[df['Flight_Duration_Minutes'] < 0, 'Flight_Duration_Minutes'] += 24 * 60
    
    # Extract delay information
    if 'ArrDelay' in df.columns:
        df['Flight_Delay'] = df['ArrDelay']
    elif 'DepDelay' in df.columns:
        df['Flight_Delay'] = df['DepDelay']
    else:
        df['Flight_Delay'] = 0
    
    # Fill missing delay values with 0
    df['Flight_Delay'] = df['Flight_Delay'].fillna(0)
    
    # Create additional features
    # Flight orientation (East-West, North-South, etc.)
    airport_data = {}
    if os.path.exists('airport_coordinates.csv'):
        airport_df = pd.read_csv('airport_coordinates.csv')
        for idx, row in airport_df.iterrows():
            airport_data[row['IATA']] = (row['Latitude'], row['Longitude'])
    
    # If we have airport location data, calculate flight orientation
    if airport_data:
        def calculate_orientation(origin, dest):
            if origin in airport_data and dest in airport_data:
                orig_lat, orig_lon = airport_data[origin]
                dest_lat, dest_lon = airport_data[dest]
                
                lat_diff = dest_lat - orig_lat
                lon_diff = dest_lon - orig_lon
                
                if abs(lat_diff) > abs(lon_diff):
                    return 'North-South'
                else:
                    return 'East-West'
            else:
                return 'Unknown'
        
        df['Orientation'] = df.apply(lambda row: calculate_orientation(row['Origin'], row['Dest']), axis=1)
    else:
        # Default orientation if we don't have airport data
        df['Orientation'] = 'Unknown'
    
    # Rename columns for consistency with the model
    rename_map = {
        'Reporting_Airline': 'Carrier_Airline',
    }
    df = df.rename(columns=rename_map)
    
    # Select final columns for output
    final_columns = [
        'Carrier_Airline', 'Tail_Number', 'Origin', 'Dest', 
        'Schedule_DateTime', 'Flight_Duration_Minutes', 'Flight_Delay',
        'Orientation'
    ]
    
    # Add any missing columns with default values
    for col in final_columns:
        if col not in df.columns:
            if col == 'Orientation':
                df[col] = 'Unknown'
            else:
                df[col] = np.nan
    
    # Save the processed data
    df[final_columns].to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    print(f"Total flights after processing: {len(df)}")
    
    return df

def create_flight_chains(df, output_dir, chain_length=3, max_time_diff_hours=24):
    """
    Create chains of consecutive flights for the same aircraft and split into train/val/test.
    
    Args:
        df: DataFrame containing processed flight data
        output_dir: Directory to save the output files
        chain_length: Minimum number of flights in a chain
        max_time_diff_hours: Maximum allowed time difference between consecutive flights
    """
    print("Creating flight chains...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Sort flights by tail number and schedule time
    df = df.sort_values(['Tail_Number', 'Schedule_DateTime'])
    
    # Create chains
    flight_chains = []
    current_tail = None
    current_chain = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Check if we're starting a new tail number
        if row['Tail_Number'] != current_tail:
            # Save previous chain if long enough
            if len(current_chain) >= chain_length:
                flight_chains.append(current_chain)
            
            # Start new chain with current flight
            current_tail = row['Tail_Number']
            current_chain = [row.to_dict()]
        else:
            # Check time difference with the last flight in the chain
            prev_flight = current_chain[-1]
            prev_time = pd.to_datetime(prev_flight['Schedule_DateTime']) + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])
            curr_time = pd.to_datetime(row['Schedule_DateTime'])
            time_diff = (curr_time - prev_time).total_seconds() / 3600  # hours
            
            if time_diff <= max_time_diff_hours:
                # Add to current chain
                current_chain.append(row.to_dict())
            else:
                # Save previous chain if long enough
                if len(current_chain) >= chain_length:
                    flight_chains.append(current_chain)
                
                # Start new chain with current flight
                current_chain = [row.to_dict()]
    
    # Save the last chain if long enough
    if len(current_chain) >= chain_length:
        flight_chains.append(current_chain)
    
    print(f"Created {len(flight_chains)} flight chains")
    
    # Extract features and labels for each chain of length 'chain_length'
    # For chains longer than 'chain_length', we'll use a sliding window
    features = []
    labels = []
    
    for chain in flight_chains:
        # For each chain, create sub-chains of length 'chain_length'
        for i in range(len(chain) - chain_length + 1):
            sub_chain = chain[i:i+chain_length]
            
            # Features: all flights in the sub-chain
            feature_dict = {
                'chain_id': i,
                'tail_number': sub_chain[0]['Tail_Number']
            }
            
            # Add features for each flight in the chain
            for j, flight in enumerate(sub_chain):
                for key, value in flight.items():
                    feature_dict[f'flight{j+1}_{key}'] = value
            
            # Add ground time between consecutive flights
            for j in range(1, len(sub_chain)):
                prev_flight = sub_chain[j-1]
                curr_flight = sub_chain[j]
                
                prev_time = pd.to_datetime(prev_flight['Schedule_DateTime']) + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])
                curr_time = pd.to_datetime(curr_flight['Schedule_DateTime'])
                ground_time = (curr_time - prev_time).total_seconds() / 60  # minutes
                
                feature_dict[f'ground_time_{j}'] = max(0, ground_time)
            
            # Label: delay of the last flight
            label = sub_chain[-1]['Flight_Delay']
            
            features.append(feature_dict)
            labels.append(label)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(features)
    feature_df['delay_label'] = labels
    
    # Add Previous Flight Delay (PFD) and Flight Time Difference (FTD)
    for i in range(2, chain_length + 1):
        feature_df[f'flight{i}_PFD'] = feature_df[f'flight{i-1}_Flight_Delay']
        
        # Flight Time Difference (scheduled departure time - previous flight's scheduled arrival time)
        feature_df[f'flight{i}_FTD'] = feature_df[f'ground_time_{i-1}']
    
    # Categorize delays for classification
    def categorize_delay(delay):
        if delay <= 0:
            return 0  # On time or early
        elif delay <= 15:
            return 1  # Slight delay
        elif delay <= 30:
            return 2  # Minor delay
        elif delay <= 60:
            return 3  # Moderate delay
        else:
            return 4  # Severe delay
    
    feature_df['delay_category'] = feature_df['delay_label'].apply(categorize_delay)
    
    # Split into train (70%), validation (15%), and test (15%) sets
    np.random.seed(42)
    
    # Shuffle indices
    indices = np.random.permutation(len(feature_df))
    train_size = int(0.7 * len(feature_df))
    val_size = int(0.15 * len(feature_df))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_df = feature_df.iloc[train_indices]
    val_df = feature_df.iloc[val_indices]
    test_df = feature_df.iloc[test_indices]
    
    # Save processed data
    feature_df.to_csv(os.path.join(output_dir, 'processed_flights_full_data.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'train_set.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation_set.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_set.csv'), index=False)
    
    # Create a DataFrame with the last event per tail number
    last_events = {}
    for chain in flight_chains:
        tail = chain[-1]['Tail_Number']
        if tail not in last_events or pd.to_datetime(chain[-1]['Schedule_DateTime']) > pd.to_datetime(last_events[tail]['Schedule_DateTime']):
            last_events[tail] = chain[-1]
    
    last_events_df = pd.DataFrame(list(last_events.values()))
    last_events_df.to_csv(os.path.join(output_dir, 'last_event_per_tail.csv'), index=False)
    
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    
    return feature_df

def main():
    parser = argparse.ArgumentParser(description='Process flight data for delay prediction')
    parser.add_argument('--input', type=str, default='data/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_0.csv',
                      help='Path to input CSV file')
    parser.add_argument('--output-dir', type=str, default='processedDataTest',
                      help='Directory to save processed data')
    parser.add_argument('--chain-length', type=int, default=3,
                      help='Number of consecutive flights in a chain')
    parser.add_argument('--max-time-diff', type=int, default=24,
                      help='Maximum time difference between consecutive flights (hours)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Process the raw flight data
    processed_file = os.path.join(args.output_dir, 'processed_flights.csv')
    df = clean_and_process_flight_data(args.input, processed_file)
    
    # Create flight chains and split into train/val/test sets
    create_flight_chains(df, args.output_dir, args.chain_length, args.max_time_diff)
    
    print("Data processing completed.")

if __name__ == "__main__":
    main()
