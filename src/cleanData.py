import pandas as pd
import numpy  as np
import os
import glob

from sklearn.model_selection import train_test_split
import datetime

# --- Configuration ---
DATA_DIR   = '../data/'
OUTPUT_DIR = '../processedData'

# Define the specific months and their corresponding seasons (ordered)
MONTH_SEASON_MAP = {
    3  : 'Spring',
    6  : 'Summer',
    9  : 'Autumn',
    12 : 'Winter'
}
FILENAME_PATTERN = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_{month}.csv"


# Mapping terms 
COLUMNS_TO_READ = [
    'FlightDate',                      # Needed for Schedule datetime
    'Reporting_Airline',               # Carrier airline
    'Flight_Number_Reporting_Airline', # Flight number
    'Tail_Number',                     # Tail number
    'Origin',                          # Origin
    'Dest',                            # Destination
    'CRSDepTime',                      # Schedule departure (time part)
    'CRSArrTime',                      # Schedule arrival (time part)
    'DepDelayMinutes',                 # Departure delay
    'ArrDelayMinutes',                 # Arrival delay
    'Cancelled',                       # Check if cancelled then delete.
    # Add other columns if needed based on data availability
]

# --- Helper Functions ---

def _parse_datetime(df, date_col, time_col):
    """Combines date and hhmm time columns into a datetime object."""
    # Format time: Zero-pad hhmm to 4 digits, handle '2400' -> '0000' (next day handled by date)
    time_str     = df[time_col].fillna(0).astype(int).astype(str).str.zfill(4)
    time_str     = time_str.replace('2400', '0000') # Replace 2400 with 0000 for parsing
    # Combine date and time strings
    datetime_str = df[date_col].astype(str) + ' ' + time_str.str[:2] + ':' + time_str.str[2:]
    # Convert to datetime, coercing errors to NaT (Not a Time)
    datetime_obj = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M', errors='coerce')

    # Optional: Handle original '2400' times by adding a day if needed.
    # This is complex as it depends on if the *date* also needs incrementing.
    # For simplicity here, we assume '2400' meant midnight of the *given* date.

    return datetime_obj

# --- Main Processing Class ---

class FlightDataProcessor:
    """Flight data processing logic"""
    def __init__(self, data_dir, file_pattern, month_map, columns_to_read):
        self.data_dir        = data_dir
        self.file_pattern    = file_pattern
        self.month_map       = month_map
        self.columns_to_read = columns_to_read
        self.raw_df          = None
        self.processed_df    = None # Holds the final dataset after FTD/PFD calc
        self.last_point_df   = None # Useful to identify last flights per tail

    def load_and_prepare_initial_data(self):
        """Loads data from multiple CSVs, assigns seasons, selects columns."""
        all_dataframes = []
        print("--- Stage 1: Loading Initial Data ---")
        for month in self.month_map.keys():
            season   = self.month_map[month]
            filename = self.file_pattern.format(month=month)
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                print(f"Reading {filename} (Season: {season})...")
                try:
                    # Read CSV, only loading necessary columns
                    # Make a copy of columns to avoid modifying the class attribute if a file lacks a column
                    cols_to_read_for_file = self.columns_to_read[:]
                    # Check which columns actually exist in the file to avoid errors
                    try:
                        file_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
                        cols_present     = [col for col in cols_to_read_for_file if col in file_cols]
                        if len(cols_present) < len(cols_to_read_for_file):
                            missing_cols = set(cols_to_read_for_file) - set(cols_present)
                            print(f"  Warning: Columns not found in {filename}: {missing_cols}. Will proceed without them.")
                            cols_to_read_for_file = cols_present
                    except Exception as e:
                         print(f"  Warning: Could not check columns for {filename}: {e}")
                         # Proceed assuming columns exist, might fail later

                    df_month           = pd.read_csv(filepath, usecols=cols_to_read_for_file, low_memory=False)
                    df_month['Season'] = season
                    all_dataframes.append(df_month)
                    print(f" -> Found {len(df_month)} records.")
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
            else:
                print(f"Warning: File not found - {filepath}. Skipping.")

        if not all_dataframes:
            print("Error: No dataframes were loaded. Exiting.")
            return False

        self.raw_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total raw records loaded: {len(self.raw_df)}")

        # Ensure all expected columns exist in the combined dataframe, fill missing ones with NaN
        for col in self.columns_to_read:
             if col not in self.raw_df.columns:
                 print(f"Warning: Column '{col}' not found in any loaded file. Adding as NaN.")
                 self.raw_df[col] = np.nan

        return True

    def preprocess_data(self):
        """Pre-processing data"""
        if self.raw_df is None:
            print("Error: Raw data not loaded.")
            return False

        print("\n--- Stage 2: Preprocessing Data ---")
        df = self.raw_df.copy()
        initial_rows = len(df)
        print(f"Initial rows: {initial_rows}")

        # Remove canceled flights
        # Important: I am Assuming 'Cancelled' == 1.0
        cancelled_count     = 0
        if 'Cancelled' in df.columns:
            # Ensure Cancelled is numeric, coercing errors
            df['Cancelled'] = pd.to_numeric(df['Cancelled'], errors='coerce')
            cancelled_count = df['Cancelled'].sum() # Sum works correctly after fillna
            df              = df[df['Cancelled'] != 1.0]
            print(f"Removed {cancelled_count:.0f} cancelled flights.")
        else:
            print("Warning: 'Cancelled' column not found, skipping cancellation filter.")

        # Identify key columns for NaN checks (adjust as needed)
        key_cols_for_nan = [
            'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
            'Tail_Number', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime',
            'DepDelayMinutes', 'ArrDelayMinutes'
        ]
        # Ensure key columns actually exist before trying to dropna
        key_cols_present = [col for col in key_cols_for_nan if col in df.columns]

        rows_before_nan  = len(df)
        df.dropna(subset=key_cols_present, inplace=True)
        rows_after_nan   = len(df)
        print(f"Removed {rows_before_nan - rows_after_nan} rows with missing values in key columns: {key_cols_present}.")

        # Convert delay columns and ElapsedTime to numeric, handling potential errors
        numeric_cols = ['DepDelayMinutes', 'ArrDelayMinutes']
        for col in numeric_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where conversion failed in these specific columns
        rows_before_num_nan = len(df)
        df.dropna(subset=[col for col in numeric_cols if col in df.columns], inplace=True)
        rows_after_num_nan = len(df)
        print(f"Removed {rows_before_num_nan - rows_after_num_nan} rows with non-numeric delay/elapsed time.")


        # Remove duplicates
        initial_rows_after_nan = len(df)
        df.drop_duplicates(inplace=True)
        print(f"Removed {initial_rows_after_nan - len(df)} duplicate rows.")

        print(f"Rows after preprocessing: {len(df)}")
        self.processed_df = df # Store preprocessed data temporarily
        return True


    def reshape_and_calculate_features(self):
        """FTD/PFD Calculation (No historical/future split based on time) """
        if self.processed_df is None: # Use data from preprocessing step
            print("Error: Data not preprocessed.")
            return False

        print("\n--- Stage 3: Reshaping Data and Calculating Features ---")
        df = self.processed_df.copy() # Start with preprocessed data

        # --- Prepare datetime columns ---
        print("Parsing schedule departure/arrival times...")
        df['Schedule_Departure_DT'] = _parse_datetime(df, 'FlightDate', 'CRSDepTime')
        df['Schedule_Arrival_DT']   = _parse_datetime(df, 'FlightDate', 'CRSArrTime')

        # Handle potential NaTs created during parsing (e.g., invalid times)
        rows_before_dt_nan          = len(df)
        df.dropna(subset=['Schedule_Departure_DT', 'Schedule_Arrival_DT'], inplace=True)
        print(f"Removed {rows_before_dt_nan - len(df)} rows with invalid schedule datetimes.")
        print(f"Rows after datetime parsing & cleanup: {len(df)}")

        # --- Create Departure and Arrival Frames and Concatenate ---
        print("Creating departure and arrival frames...")
        # Common columns to duplicate
        common_cols = [
            'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Tail_Number',
            'Origin', 'Dest', 'Season'
        ]
        # Ensure all common columns exist in df
        common_cols = [col for col in common_cols if col in df.columns]

        # Departure frame
        dep_cols    = common_cols + ['Schedule_Departure_DT', 'DepDelayMinutes']
        dep_df      = df[[col for col in dep_cols if col in df.columns]].copy()
        dep_df['Orientation'] = 'Departure'
        dep_df.rename(columns={
            'Schedule_Departure_DT' : 'Schedule_DateTime', # Consistent naming
            'DepDelayMinutes'       : 'Flight_Delay'
        }, inplace=True)

        # Arrival frame
        arr_cols              = common_cols + ['Schedule_Arrival_DT', 'ArrDelayMinutes']
        arr_df                = df[[col for col in arr_cols if col in df.columns]].copy()
        arr_df['Orientation'] = 'Arrival'
        arr_df.rename(columns={
            'Schedule_Arrival_DT' : 'Schedule_DateTime', # Consistent naming
            'ArrDelayMinutes'     : 'Flight_Delay'
        }, inplace=True)

        # Concatenate departure and arrival frames
        print("Concatenating frames...")
        flight_df = pd.concat([dep_df, arr_df], ignore_index=True)

        # Rename columns for clarity
        flight_df.rename(columns={
            'Reporting_Airline'               : 'Carrier_Airline',
            'Flight_Number_Reporting_Airline' : 'Flight_Number',
            'Tail_Number'                     : 'Tail_Number',
            'Schedule_DateTime'               : 'Schedule_DateTime',
        }, inplace=True)

        # Ensure correct data types for delays and elapsed time
        flight_df['Flight_Delay'] = pd.to_numeric(flight_df['Flight_Delay'], errors='coerce').fillna(0)
        if 'CRS_Elapsed_Time' in flight_df.columns:
            flight_df['CRS_Elapsed_Time'] = pd.to_numeric(flight_df['CRS_Elapsed_Time'], errors='coerce').fillna(0)

        print(f"Reshaped data frame ('Flight') created with {len(flight_df)} rows.")

        # --- Initialize FTD and PFD ---
        flight_df['FTD'] = 0.0 # Flight Time Duration (time since last landing/takeoff for this tail) in minutes
        flight_df['PFD'] = 0.0 # Previous Flight Delay in minutes

        # --- FTD Calculation ---
        print("Calculating Flight Time Duration (FTD - time between consecutive events for a tail)...")
        # Sort by tail number and time to properly calculate differences
        flight_df.sort_values(by=['Tail_Number', 'Schedule_DateTime'], inplace=True)
        flight_df.reset_index(drop=True, inplace=True) # Reset index after sort

        # Calculate time difference between consecutive events *for the same tail number*
        flight_df['FTD_Timedelta'] = flight_df.groupby('Tail_Number')['Schedule_DateTime'].diff()

        # Convert Timedelta to minutes, fill NaT (first event for each tail) with 0
        flight_df['FTD'] = flight_df['FTD_Timedelta'].dt.total_seconds() / 60.0
        flight_df['FTD'] = flight_df['FTD'].fillna(0.0) # Fill NaNs for first events
        flight_df.drop(columns=['FTD_Timedelta'], inplace=True) # Remove temporary column

        # --- PFD Calculation ---
        # Since there's no time cutoff, we calculate PFD for the entire dataset.
        print("Calculating Previous Flight Delay (PFD) for all data...")

        # Get the 'Flight_Delay' from the previous row *for the same tail number*
        flight_df['PFD'] = flight_df.groupby('Tail_Number')['Flight_Delay'].shift(1)

        # Fill NaN PFD values (first event for each tail number) with 0
        flight_df['PFD'] = flight_df['PFD'].fillna(0.0)

        # --- Last Point Logic (Identify last event for each tail number in the dataset) ---
        print("Identifying last recorded event for each tail number...")
        # Find rows where the next row has a different tail number or is the end of the dataframe
        is_last = (flight_df['Tail_Number'] != flight_df['Tail_Number'].shift(-1))
        self.last_point_df = flight_df[is_last].copy()
        print(f"Found {len(self.last_point_df)} last points (last event per tail).")

        # Assign the final processed df
        self.processed_df = flight_df
        print("\n--- Stage 3 Complete ---")
        return True


    def split_and_sample_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """Performs Train/Validation/Test splits and subsampling experiments on the processed data."""
        if self.processed_df is None or self.processed_df.empty:
            print("Error: No processed data available for splitting.")
            return None, None, None

        print("\n--- Stage 4: Splitting Data and Subsampling Experiments ---")
        data_to_split = self.processed_df # Use the fully processed data

        print(f"Splitting data (Test={test_size*100}%, Validation={val_size*100}% of remainder)...")
        # Split into Train+Validation and Test
        train_val_df, test_df = train_test_split(
            data_to_split,
            test_size    = test_size,
            random_state = random_state,
        )

        # ---- Split Train+Validation into Train and Validation ----

        # Calculate effective validation size relative to train_val_df
        relative_val_size = val_size # val_size is relative to train_val_df now
        train_df, val_df  = train_test_split(
            train_val_df,
            test_size     = relative_val_size, # e.g. 0.25 means 25% of train_val_df -> validation
            random_state  = random_state
        )

        print(f"Train set size      : {len(train_df)}")
        print(f"Validation set size : {len(val_df)}")
        print(f"Test set size       : {len(test_df)}")

        # --- Subsampling Experiment 1: Varying Training Data Size ---
        print("\n--- Subsampling Experiment 1: Varying Training Data Size ---")
        train_percentages = [20, 40, 60, 80, 100] 
        n_runs = 5 # Honestly random choice lol

        for percent in train_percentages:
            print(f"\n  Training with {percent}% of training data ({n_runs} runs):")
            delay_stats = []
            if train_df.empty:
                print("    Train DataFrame is empty, cannot sample.")
                continue
            for i in range(n_runs):
                # Ensure sample size isn't zero if percent is small
                frac = percent / 100.0
                if int(len(train_df) * frac) == 0 and frac > 0:
                    print(f"    Skipping run {i+1}: Sample size would be zero.")
                    continue
                sample_df = train_df.sample(frac=frac, random_state=random_state + i)
                if sample_df.empty:
                     print(f"    Run {i+1}: Sample is empty.")
                     continue
                # In a real scenario, train model on 'sample_df', evaluate on 'val_df'
                mean_delay = sample_df['Flight_Delay'].mean()
                std_delay  = sample_df['Flight_Delay'].std()
                delay_stats.append(mean_delay)
                print(f"    Run {i+1}: Sample size={len(sample_df)}, Mean Delay={mean_delay:.2f}, Std Delay={std_delay:.2f}")

            if delay_stats:
                 print(f"  -> Avg Mean Delay across runs: {np.mean(delay_stats):.2f}, Std Dev of Mean Delays: {np.std(delay_stats):.2f}")

        # --- Subsampling Experiment 2: Varying Validation Data Size ---
        print("\n--- Subsampling Experiment 2: Varying Validation Data Size ---")
        # Conceptually train on all training data
        print(f"  (Conceptual) Training on full training set (size={len(train_df)}).")
        val_percentages = [20, 40, 60, 80, 100]
        n_runs = 5

        for percent in val_percentages:
            print(f"\n  Evaluating on {percent}% of validation data ({n_runs} runs):")
            delay_stats = []
            if val_df.empty:
                print("    Validation DataFrame is empty, cannot sample.")
                continue
            for i in range(n_runs):
                 frac = percent/100.0
                 if int(len(val_df) * frac) == 0 and frac > 0:
                     print(f"    Skipping run {i+1}: Sample size would be zero.")
                     continue
                 sample_val_df = val_df.sample(frac=frac, random_state=random_state + i)
                 if sample_val_df.empty:
                     print(f"    Run {i+1}: Validation sample is empty.")
                     continue
                 # Evaluate performance (e.g., calculate metrics) on sample_val_df
                 mean_delay = sample_val_df['Flight_Delay'].mean()
                 std_delay = sample_val_df['Flight_Delay'].std()
                 delay_stats.append(mean_delay)
                 print(f"    Run {i+1}: Validation sample size={len(sample_val_df)}, Mean Delay={mean_delay:.2f}, Std Delay={std_delay:.2f}")

            if delay_stats:
                 print(f"  -> Avg Mean Delay across runs: {np.mean(delay_stats):.2f}, Std Dev of Mean Delays: {np.std(delay_stats):.2f}")

        print("\n--- Stage 4 Complete ---")
        return train_df, val_df, test_df


    def save_results(self, train_df, val_df, test_df, output_dir):
        """Saves the processed dataframes to CSV files."""
        print(f"\n--- Stage 5: Saving Results to '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save the main processed dataframe (contains all events with FTD/PFD)
            if self.processed_df is not None and not self.processed_df.empty:
                 # Changed filename to reflect it contains all data now
                 path = os.path.join(output_dir, "processed_flights_full_data.csv")
                 self.processed_df.to_csv(path, index=False)
                 print(f"Saved fully processed data to {path}")

            # Removed saving of 'future_df' as it's not relevant without the time split

            # Save the last points identified
            if self.last_point_df is not None and not self.last_point_df.empty:
                 path = os.path.join(output_dir, "last_event_per_tail.csv")
                 self.last_point_df.to_csv(path, index=False)
                 print(f"Saved last event per tail number to {path}")
            else:
                 print("No last points identified/saved.")

            # Save splits
            if train_df is not None and not train_df.empty:
                 path = os.path.join(output_dir, "train_set.csv")
                 train_df.to_csv(path, index=False)
                 print(f"Saved train set to {path}")
            else:
                 print("Train set is empty, not saved.")

            if val_df is not None and not val_df.empty:
                 path = os.path.join(output_dir, "validation_set.csv")
                 val_df.to_csv(path, index=False)
                 print(f"Saved validation set to {path}")
            else:
                print("Validation set is empty, not saved.")

            if test_df is not None and not test_df.empty:
                 path = os.path.join(output_dir, "test_set.csv")
                 test_df.to_csv(path, index=False)
                 print(f"Saved test set to {path}")
            else:
                print("Test set is empty, not saved.")


        except Exception as e:
            print(f"Error saving results: {e}")

        print("\n--- Stage 5 Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Flight Data Processing Pipeline...")

    processor = FlightDataProcessor(
        data_dir        = DATA_DIR,
        file_pattern    = FILENAME_PATTERN,
        month_map       = MONTH_SEASON_MAP,
        columns_to_read = COLUMNS_TO_READ
    )

    # Run the pipeline steps
    if processor.load_and_prepare_initial_data():
        if processor.preprocess_data():
             # Pass no cutoff_time argument now
            if processor.reshape_and_calculate_features():
                train_df, val_df, test_df = processor.split_and_sample_data()
                processor.save_results(train_df, val_df, test_df, output_dir=OUTPUT_DIR)

    print("\nProcessing Pipeline Finished.")
