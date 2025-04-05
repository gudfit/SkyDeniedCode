# uv run cleanRealTime.py
import pandas as pd
import numpy  as np
import os
import glob
import datetime

from sklearn.model_selection import train_test_split


# --- Configuration ---
# Keep your existing DATA_DIR, OUTPUT_DIR, MONTH_SEASON_MAP, FILENAME_PATTERN
DATA_DIR         = '../data/'
OUTPUT_DIR       = '../processedData' # Output for this script

# Define the specific months and their corresponding seasons (ordered)
MONTH_SEASON_MAP = {
    3  : 'Spring',
    6  : 'Summer',
    9  : 'Autumn',
    12 : 'Winter',
    0  : 'Test', # For quick testing if needed
}

FILENAME_PATTERN  = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_{month}.csv"

COLUMNS_TO_READ   = [
    'FlightDate',
    'Reporting_Airline',
    'Flight_Number_Reporting_Airline',
    'Tail_Number',
    'Origin',
    'Dest',
    'CRSDepTime',
    'CRSArrTime',
    'DepDelayMinutes',
    'ArrDelayMinutes',
    'Cancelled',
]

# --- Helper Functions ---

def _parse_datetime(df, date_col, time_col):
    """
    Combines date and hhmm time columns into a datetime object.
    Handles '2400' by parsing it as '0000'. The date increment for overnight
    flights is handled later in the main processing logic.
    """
    # Format time: Zero-pad hhmm to 4 digits
    time_float   = pd.to_numeric(df[time_col], errors='coerce').fillna(0)
    time_str     = time_float.astype(int).astype(str).str.zfill(4)
    time_str     = time_str.replace('2400', '0000')

    # Combine date and time strings
    datetime_str = df[date_col].astype(str) + ' ' + time_str.str[:2] + ':' + time_str.str[2:]

    # Convert to datetime, coercing errors to NaT (Not a Time)
    datetime_obj = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M', errors='coerce')

    return datetime_obj

# --- Main Processing Class ---

class FlightDataProcessor:
    """Encapsulates the flight data processing logic."""
    def __init__(self, data_dir, file_pattern, month_map, columns_to_read):
        self.data_dir        = data_dir
        self.file_pattern    = file_pattern
        self.month_map       = month_map
        self.columns_to_read = columns_to_read
        self.raw_df          = None
        self.processed_df    = None # Holds data after preprocessing, before reshape
        self.final_flight_df = None # Holds the final reshaped data with InterEventTimeMinutes/PFD
        self.last_point_df   = None # Holds last event per tail

    def load_and_prepare_initial_data(self):
        """Loads data from multiple CSVs, assigns seasons, selects columns."""
        all_dataframes                            = []
        print("--- Stage 1: Loading Initial Data ---")
        required_cols_for_parse                   = ['FlightDate', 'CRSDepTime', 'CRSArrTime']
        if not all(col in self.columns_to_read for col in required_cols_for_parse):
             print(f"Error: Essential columns for datetime parsing {required_cols_for_parse} are missing from COLUMNS_TO_READ.")
             return False

        for month in self.month_map.keys():
            season                                = self.month_map[month]
            filename                              = self.file_pattern.format(month=month)
            filepath                              = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                print(f"Reading {filename} (Season: {season})...")
                try:
                    cols_to_read_for_file         = self.columns_to_read[:]
                    try:
                        file_cols                 = pd.read_csv(filepath, nrows=0).columns.tolist()
                        cols_present              = [col for col in cols_to_read_for_file if col in file_cols]
                        if len(cols_present)      < len(cols_to_read_for_file):
                            missing_cols          = set(cols_to_read_for_file) - set(cols_present)
                            print(f"  Warning: Columns not found in {filename}: {missing_cols}. Will proceed without them.")
                            cols_to_read_for_file = cols_present
                    except Exception as e:
                         print(f"  Warning: Could not check columns for {filename}: {e}")

                    if not all(col in cols_to_read_for_file for col in required_cols_for_parse):
                        print(f"  Error: Essential datetime columns missing in file {filename} after check. Skipping file.")
                        continue

                    df_month                      = pd.read_csv(filepath, usecols=cols_to_read_for_file, low_memory=False)
                    df_month['Season']            = season
                    all_dataframes.append(df_month)
                    print(f" -> Found {len(df_month)} records.")
                except ValueError as ve:
                    print(f"Error reading file {filename}: {ve}. Check column names.")
                except Exception as e:
                    print(f"General error reading file {filename}: {e}")
            else:
                print(f"Warning: File not found - {filepath}. Skipping.")

        if not all_dataframes:
            print("Error: No dataframes were loaded or essential columns missing. Exiting.")
            return False

        self.raw_df                               = pd.concat(all_dataframes, ignore_index=True)
        print(f"Total raw records loaded: {len(self.raw_df)}")

        for col in self.columns_to_read:
             if col not in self.raw_df.columns:
                 print(f"Warning: Column '{col}' was specified but not found in any loaded file. Adding as NaN.")
                 self.raw_df[col]                 = np.nan

        return True

    def preprocess_data(self):
        """Pre-processing: remove canceled, missing values, duplicates"""
        if self.raw_df is None:
            print("Error: Raw data not loaded.")
            return False

        print("\n--- Stage 2: Preprocessing Data ---")
        df                          = self.raw_df.copy()
        initial_rows                = len(df)
        print(f"Initial rows: {initial_rows}")

        cancelled_count             = 0
        if 'Cancelled' in df.columns:
            df['Cancelled']         = pd.to_numeric(df['Cancelled'], errors='coerce')
            # Ensure we handle potential NaNs from coercion before summing
            cancelled_rows          = df[df['Cancelled'] == 1.0]
            cancelled_count         = len(cancelled_rows)
            df                      = df[df['Cancelled'] != 1.0] # Keep only non-cancelled
            print(f"Removed {cancelled_count} cancelled flights.")
        else:
            print("Warning: 'Cancelled' column not found, skipping cancellation filter.")

        key_cols_for_nan           = [
            'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline',
            'Tail_Number', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime',
            'DepDelayMinutes', 'ArrDelayMinutes'
        ]
        key_cols_present           = [col for col in key_cols_for_nan if col in df.columns]
        missing_key_cols           = set(key_cols_for_nan) - set(key_cols_present)
        if missing_key_cols:
            print(f"Warning: Key columns for NaN check missing: {missing_key_cols}. Proceeding without them for NaN drop.")

        rows_before_nan            = len(df)
        if key_cols_present:
            # Also drop rows where Tail_Number might be specifically missing/problematic like '000000' if that occurs
            if 'Tail_Number' in df.columns:
                 df = df[df['Tail_Number'].notna() & (df['Tail_Number'] != '000000')] # Example invalid value
            df.dropna(subset       = key_cols_present, inplace=True)

        rows_after_nan             = len(df)
        removed_nan_count          = rows_before_nan - rows_after_nan
        print(f"Removed {removed_nan_count} rows with missing values in key columns: {key_cols_present} (and potentially invalid Tail_Number).")

        numeric_cols               = ['DepDelayMinutes', 'ArrDelayMinutes', 'CRSDepTime', 'CRSArrTime']
        for col in numeric_cols:
            if col in df.columns:
                 df[col]           = pd.to_numeric(df[col], errors='coerce')

        rows_before_num_nan        = len(df)
        numeric_cols_present       = [col for col in numeric_cols if col in df.columns]
        if numeric_cols_present:
            df.dropna(subset       = numeric_cols_present, inplace=True)
        rows_after_num_nan         = len(df)
        removed_num_nan            = rows_before_num_nan - rows_after_num_nan
        print(f"Removed {removed_num_nan} rows with non-numeric values in essential time/delay columns.")

        initial_rows_after_clean   = len(df)
        df.drop_duplicates(inplace = True)
        removed_duplicates         = initial_rows_after_clean - len(df)
        print(f"Removed {removed_duplicates} duplicate rows.")
        print(f"Rows after preprocessing: {len(df)}")
        self.processed_df          = df
        return True


    def reshape_and_calculate_features(self):
        """InterEventTimeMinutes/PFD Calculation, calculating elapsed time manually. """
        if self.processed_df is None or self.processed_df.empty:
            print("Error: Data not preprocessed or is empty.")
            return False

        print("\n--- Stage 3: Reshaping Data and Calculating Features ---")
        df = self.processed_df.copy()

        print("Parsing schedule departure/arrival times...")
        df['Schedule_Departure_DT'] = _parse_datetime(df, 'FlightDate', 'CRSDepTime')
        df['Schedule_Arrival_DT']   = _parse_datetime(df, 'FlightDate', 'CRSArrTime')

        print("Correcting arrival dates for overnight flights (including '2400')...")
        rows_before_dt_nan = len(df)
        df.dropna(subset   = ['Schedule_Departure_DT', 'Schedule_Arrival_DT'], inplace=True)
        removed_dt_nan     = rows_before_dt_nan - len(df)
        if removed_dt_nan > 0:
            print(f"Removed {removed_dt_nan} rows with invalid schedule datetimes after initial parse.")

        if df.empty:
            print("Error: No valid data remaining after initial datetime parsing.")
            return False

        overnight_condition = df['Schedule_Arrival_DT'] <= df['Schedule_Departure_DT']
        num_overnight       = overnight_condition.sum()
        print(f" -> Identified {num_overnight} potential overnight/midnight arrivals requiring date adjustment.")
        if num_overnight > 0:
            df.loc[overnight_condition, 'Schedule_Arrival_DT'] = df.loc[overnight_condition, 'Schedule_Arrival_DT'] + pd.Timedelta(days=1)
            print(" -> Arrival dates adjusted.")

        print("Calculating flight duration from corrected schedule times...")
        valid_times                                        = df['Schedule_Arrival_DT'].notna() & df['Schedule_Departure_DT'].notna()
        df['Calculated_Duration_Minutes']                  = np.nan
        df.loc[valid_times, 'Calculated_Duration_Minutes'] = \
            (df.loc[valid_times, 'Schedule_Arrival_DT'] - df.loc[valid_times, 'Schedule_Departure_DT']).dt.total_seconds() / 60.0

        invalid_duration_count = (df['Calculated_Duration_Minutes'] <= 0).sum()
        if invalid_duration_count > 0:
             print(f"Warning: Found {invalid_duration_count} rows with non-positive calculated flight duration. Removing them.")
             df = df[df['Calculated_Duration_Minutes'] > 0] # Remove non-positive durations

        rows_before_dur_nan = len(df)
        df.dropna(subset    = ['Calculated_Duration_Minutes'], inplace=True)
        removed_dur_nan     = rows_before_dur_nan - len(df)
        if removed_dur_nan  > 0:
             print(f"Removed {removed_dur_nan} rows with invalid calculated duration.")

        print(f"Rows after datetime processing & duration calculation: {len(df)}")
        if len(df) == 0:
            print("Error: No valid data remaining after duration calculation.")
            return False

        print("Creating departure and arrival frames...")
        common_cols_base = [
            'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Tail_Number',
            'Origin', 'Dest', 'Season', 'Calculated_Duration_Minutes'
        ]
        common_cols    = [col for col in common_cols_base if col in df.columns]
        missing_common = set(common_cols_base) - set(common_cols)
        if missing_common:
             print(f"Warning: Columns missing for reshape: {missing_common}. They won't be in the final dataset.")

        # Use Calculated_Duration_Minutes consistently
        cols_for_dep = common_cols + ['Schedule_Departure_DT', 'DepDelayMinutes']
        cols_for_arr = common_cols + ['Schedule_Arrival_DT', 'ArrDelayMinutes']

        dep_df = df[[col for col in cols_for_dep if col in df.columns]].copy()
        dep_df['Orientation'] = 'Departure'
        dep_df.rename(columns={
            'Schedule_Departure_DT'           : 'Schedule_DateTime',
            'DepDelayMinutes'                 : 'Flight_Delay',
            'Reporting_Airline'               : 'Carrier_Airline',
            'Flight_Number_Reporting_Airline' : 'Flight_Number',
            'Calculated_Duration_Minutes'     : 'Flight_Duration_Minutes' # Consistent name
        }, inplace=True)

        arr_df = df[[col for col in cols_for_arr if col in df.columns]].copy()
        arr_df['Orientation'] = 'Arrival'
        arr_df.rename(columns={
            'Schedule_Arrival_DT'             : 'Schedule_DateTime',
            'ArrDelayMinutes'                 : 'Flight_Delay',
            'Reporting_Airline'               : 'Carrier_Airline',
            'Flight_Number_Reporting_Airline' : 'Flight_Number',
            'Calculated_Duration_Minutes'     : 'Flight_Duration_Minutes' # Consistent name
        }, inplace=True)

        print("Concatenating frames...")
        flight_df = pd.concat([dep_df, arr_df], ignore_index=True)

        flight_df['Flight_Delay'] = pd.to_numeric(flight_df['Flight_Delay'], errors='coerce').fillna(0.0)
        if 'Flight_Duration_Minutes' in flight_df.columns:
            flight_df['Flight_Duration_Minutes'] = pd.to_numeric(flight_df['Flight_Duration_Minutes'], errors='coerce').fillna(0.0)

        print(f"Reshaped data frame ('Flight') created with {len(flight_df)} rows.")
        print(f"Columns in reshaped frame: {flight_df.columns.tolist()}")

        if flight_df.empty:
            print("Error: Reshaped DataFrame is empty. Cannot calculate features.")
            return False

        # --- Initialize InterEventTimeMinutes and PFD ---
        # *** RENAME FTD ***
        flight_df['InterEventTimeMinutes'] = 0.0 # Time since last event for this tail
        flight_df['PFD'] = 0.0 # Previous Flight Delay

        # --- InterEventTimeMinutes Calculation ---
        # *** RENAME FTD ***
        print("Calculating InterEventTimeMinutes (time between consecutive events for a tail)...")
        # Ensure sorting is robust, especially if times are identical
        flight_df.sort_values(by=['Tail_Number', 'Schedule_DateTime', 'Orientation'], inplace=True) # Add Orientation as tie-breaker
        flight_df.reset_index(drop=True, inplace=True)

        # Calculate time difference in minutes
        time_diff_seconds = flight_df.groupby('Tail_Number')['Schedule_DateTime'].diff().dt.total_seconds()
        flight_df['InterEventTimeMinutes'] = (time_diff_seconds / 60.0).fillna(0.0)

        # Optional: Sanity check for negative or zero intervals after diff (shouldn't happen with proper sorting/cleaning)
        neg_interval_count = (flight_df['InterEventTimeMinutes'] < 0).sum()
        if neg_interval_count > 0:
            print(f"Warning: Found {neg_interval_count} negative InterEventTimeMinutes values. Check sorting and data.")
            # Decide how to handle: set to 0, remove row, or investigate
            flight_df.loc[flight_df['InterEventTimeMinutes'] < 0, 'InterEventTimeMinutes'] = 0.0


        # --- PFD Calculation ---
        print("Calculating Previous Flight Delay (PFD) for all data...")
        # Shift needs to happen *after* the correct sorting
        flight_df['PFD'] = flight_df.groupby('Tail_Number')['Flight_Delay'].shift(1).fillna(0.0)

        # --- Last Point Logic ---
        # This can remain the same, it just identifies the last row per tail in the sorted df
        print("Identifying last recorded event for each tail number...")
        # Shift(-1) looks at the *next* row. The condition is True if the next Tail_Number is different OR if it's the very last row.
        is_last = (flight_df['Tail_Number'].shift(-1) != flight_df['Tail_Number'])
        self.last_point_df = flight_df[is_last].copy()
        print(f"Found {len(self.last_point_df)} last points (last event per tail).")


        self.final_flight_df = flight_df
        print("\n--- Stage 3 Complete ---")
        print(f"Columns after Stage 3: {self.final_flight_df.columns.tolist()}")
        return True

    # --- split_and_sample_data ---
    # Keep this function as is for splitting and analysis if needed.
    # It operates on self.final_flight_df which now has the corrected features.
    def split_and_sample_data(self, test_size=0.2, val_size=0.25, random_state=42):
        """Performs Train/Validation/Test splits and subsampling experiments."""
        if self.final_flight_df is None or self.final_flight_df.empty:
            print("Error: No final processed data available for splitting.")
            return None, None, None

        print("\n--- Stage 4: Splitting Data and Subsampling Experiments ---")
        data_to_split = self.final_flight_df

        # Check if required columns exist before proceeding
        required_split_cols = ['Tail_Number', 'Schedule_DateTime', 'Orientation', 'Flight_Delay']
        if not all(col in data_to_split.columns for col in required_split_cols):
            print(f"Error: Missing one or more required columns for splitting: {required_split_cols}")
            return None, None, None

        print(f"Splitting data (Test={test_size*100}%, Validation={val_size*100}% of remainder)...")

        stratify_col = None
        if 'Season' in data_to_split.columns and data_to_split['Season'].nunique() > 1:
             stratify_col = 'Season'
             print(f"Attempting to stratify split by '{stratify_col}'.")
        elif 'Orientation' in data_to_split.columns and data_to_split['Orientation'].nunique() > 1:
             stratify_col = 'Orientation'
             print(f"Attempting to stratify split by '{stratify_col}'.")
        else:
             print("No suitable column found for stratification or column has only one value.")

        stratify_data = data_to_split[stratify_col] if stratify_col else None

        try:
            # Ensure stratify_data has the same index as data_to_split before passing
            if stratify_data is not None:
                stratify_data = stratify_data.loc[data_to_split.index]

            train_val_df, test_df = train_test_split(
                data_to_split,
                test_size    = test_size,
                random_state = random_state,
                stratify     = stratify_data
            )
        except ValueError as e:
             print(f"Warning: Stratified split failed ({e}). Performing non-stratified split.")
             stratify_data         = None # Disable stratification for subsequent splits too
             train_val_df, test_df = train_test_split(
                data_to_split,
                test_size     = test_size,
                random_state  = random_state
             )

        # Ensure stratify_train_val is aligned with train_val_df
        stratify_train_val = train_val_df[stratify_col] if stratify_col and stratify_data is not None else None
        if stratify_train_val is not None:
            stratify_train_val = stratify_train_val.loc[train_val_df.index]


        try:
             train_df, val_df = train_test_split(
                 train_val_df,
                 test_size    = val_size, # val_size is relative to train_val_df
                 random_state = random_state,
                 stratify     = stratify_train_val
             )
        except ValueError as e:
             print(f"Warning: Stratified split for train/validation failed ({e}). Performing non-stratified split.")
             train_df, val_df = train_test_split(
                 train_val_df,
                 test_size    = val_size,
                 random_state = random_state
             )

        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")

        # --- Subsampling Experiments (Keep as is) ---
        print("\n--- Subsampling Experiment 1: Varying Training Data Size ---")
        train_percentages = [20, 40, 60, 80, 100]
        n_runs = 5 # Keep runs low for speed

        for percent in train_percentages:
            print(f"\n  Training with {percent}% of training data ({n_runs} runs):")
            delay_stats = []
            if train_df.empty:
                print("    Train DataFrame is empty, cannot sample.")
                continue
            frac = percent / 100.0
            if int(len(train_df) * frac) == 0 and frac > 0:
                 print(f"    Skipping {percent}% runs: Sample size would be zero.")
                 continue

            for i in range(n_runs):
                # Ensure Flight_Delay exists before sampling
                if 'Flight_Delay' not in train_df.columns:
                    print("    Error: 'Flight_Delay' column missing in train_df.")
                    break
                sample_df  = train_df.sample(frac=frac, random_state=random_state + i)
                if sample_df.empty:
                     print(f"    Run {i+1}: Sample is empty (frac={frac}).")
                     continue
                mean_delay = sample_df['Flight_Delay'].mean()
                std_delay  = sample_df['Flight_Delay'].std()
                delay_stats.append(mean_delay)
                print(f"    Run {i+1}: Sample size={len(sample_df)}, Mean Delay={mean_delay:.2f}, Std Delay={std_delay:.2f}")

            if delay_stats:
                 print(f"  -> Avg Mean Delay across runs: {np.mean(delay_stats):.2f}, Std Dev of Mean Delays: {np.std(delay_stats):.2f}")
            elif not train_df.empty and 'Flight_Delay' in train_df.columns :
                 print("  -> No valid runs completed for this percentage.")
            elif 'Flight_Delay' not in train_df.columns:
                 print(" -> Cannot run experiment, Flight_Delay column missing.")


        print("\n--- Subsampling Experiment 2: Varying Validation Data Size ---")
        print(f"  (Conceptual) Training on full training set (size={len(train_df)}).")
        val_percentages = [20, 40, 60, 80, 100]
        n_runs          = 5

        for percent in val_percentages:
            print(f"\n  Evaluating on {percent}% of validation data ({n_runs} runs):")
            delay_stats                = []
            if val_df.empty:
                print("    Validation DataFrame is empty, cannot sample.")
                continue
            frac                       = percent / 100.0
            if int(len(val_df) * frac) == 0 and frac > 0:
                 print(f"    Skipping {percent}% runs: Sample size would be zero.")
                 continue

            for i in range(n_runs):
                 if 'Flight_Delay' not in val_df.columns:
                     print("    Error: 'Flight_Delay' column missing in val_df.")
                     break
                 sample_val_df = val_df.sample(frac=frac, random_state=random_state + i)
                 if sample_val_df.empty:
                     print(f"    Run {i+1}: Validation sample is empty (frac={frac}).")
                     continue
                 mean_delay    = sample_val_df['Flight_Delay'].mean()
                 std_delay     = sample_val_df['Flight_Delay'].std()
                 delay_stats.append(mean_delay)
                 print(f"    Run {i+1}: Validation sample size={len(sample_val_df)}, Mean Delay={mean_delay:.2f}, Std Delay={std_delay:.2f}")

            if delay_stats:
                 print(f"  -> Avg Mean Delay across runs: {np.mean(delay_stats):.2f}, Std Dev of Mean Delays: {np.std(delay_stats):.2f}")
            elif not val_df.empty and 'Flight_Delay' in val_df.columns:
                 print("  -> No valid runs completed for this percentage.")
            elif 'Flight_Delay' not in val_df.columns:
                print(" -> Cannot run experiment, Flight_Delay column missing.")


        print("\n--- Stage 4 Complete ---")
        return train_df, val_df, test_df


    def save_results(self, train_df, val_df, test_df, output_dir):
        """Saves the processed dataframes (with InterEventTimeMinutes/PFD) to CSV files."""
        data_exists = any([
            self.final_flight_df is not None and not self.final_flight_df.empty,
            self.last_point_df   is not None and not self.last_point_df.empty,
            train_df             is not None and not train_df.empty,
            val_df               is not None and not val_df.empty,
            test_df              is not None and not test_df.empty
        ])

        if not data_exists:
             print("\n--- Stage 5: Saving Results ---")
             print("No dataframes available to save.")
             return

        print(f"\n--- Stage 5: Saving Results to '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)
        float_fmt = '%.2f' # Keep precision reasonable

        try:
            if self.final_flight_df is not None and not self.final_flight_df.empty:
                 # *** Filename reflects content ***
                 path = os.path.join(output_dir, "processed_event_data.csv")
                 self.final_flight_df.to_csv(path, index=False, float_format=float_fmt)
                 print(f"Saved processed event data to {path}")
            else:
                 print("Processed event data is empty or None, not saved.")

            if self.last_point_df is not None and not self.last_point_df.empty:
                 path = os.path.join(output_dir, "last_event_per_tail.csv")
                 self.last_point_df.to_csv(path, index=False, float_format=float_fmt)
                 print(f"Saved last event per tail number to {path}")
            else:
                 print("Last points data is empty or None, not saved.")

            if train_df is not None and not train_df.empty:
                 path = os.path.join(output_dir, "train_set.csv")
                 train_df.to_csv(path, index=False, float_format=float_fmt)
                 print(f"Saved train set to {path}")
            else:
                 print("Train set is empty or None, not saved.")

            if val_df is not None and not val_df.empty:
                 path = os.path.join(output_dir, "validation_set.csv")
                 val_df.to_csv(path, index=False, float_format=float_fmt)
                 print(f"Saved validation set to {path}")
            else:
                print("Validation set is empty or None, not saved.")

            if test_df is not None and not test_df.empty:
                 path = os.path.join(output_dir, "test_set.csv")
                 test_df.to_csv(path, index=False, float_format=float_fmt)
                 print(f"Saved test set to {path}")
            else:
                print("Test set is empty or None, not saved.")

        except Exception as e:
            print(f"Error saving results: {e}")

        print("\n--- Stage 5 Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Flight Data Processing Pipeline (Base Features)...")
    start_time = datetime.datetime.now()

    processor = FlightDataProcessor(
        data_dir        = DATA_DIR,
        file_pattern    = FILENAME_PATTERN,
        month_map       = MONTH_SEASON_MAP,
        columns_to_read = COLUMNS_TO_READ
    )

    pipeline_successful = False
    if processor.load_and_prepare_initial_data():
        if processor.preprocess_data():
            if processor.reshape_and_calculate_features():
                # Splitting is optional here, main goal is the processed_event_data.csv
                # You can choose to run split_and_sample_data or not
                # train_df, val_df, test_df = processor.split_and_sample_data()
                # For saving just the main file:
                processor.save_results(None, None, None, output_dir=OUTPUT_DIR)
                pipeline_successful = True # Mark success if saving starts
            else:
                print("Skipping subsequent steps as feature reshaping failed.")
        else:
            print("Skipping subsequent steps as preprocessing failed.")
    else:
        print("Skipping subsequent steps as data loading failed.")

    end_time = datetime.datetime.now()
    status   = "Finished Successfully" if pipeline_successful else "Finished with Errors/Warnings or Incomplete"
    print(f"\nBase Feature Processing Pipeline {status}. Total time: {end_time - start_time}")
