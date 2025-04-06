import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse # Ensure argparse is imported
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
# Default name for the coordinates file (will be combined with relative path later)
DEFAULT_COORDS_FILENAME = 'iata-icao.csv'
DEFAULT_RAW_FLIGHT_FILENAME = 'report_carrier_top10.csv'

# --- Airport Data Loading ---
def load_airport_data(filepath):
    """Loads airport coordinate data from the iata-icao.csv file."""
    airport_data = {}
    if not os.path.exists(filepath):
        print(f"Warning: Airport coordinates file not found at '{filepath}'. Orientation calculation will be skipped.")
        return None # Indicate failure to load

    try:
        print(f"Loading airport data from: {filepath}")
        # Specify dtype for IATA to avoid mixed type warnings if some look numeric
        airport_df = pd.read_csv(filepath, low_memory=False, dtype={'iata': str})

        # --- Use the columns from iata-icao.csv ---
        required_cols = ['iata', 'latitude', 'longitude']
        if not all(col in airport_df.columns for col in required_cols):
            print(f"Error: Airport coordinates file '{filepath}' is missing required columns: {required_cols}. Needs 'iata', 'latitude', 'longitude'. Orientation calculation skipped.")
            return None # Indicate failure due to format

        processed_count = 0
        skipped_count = 0
        for idx, row in airport_df.iterrows():
            # --- Use the correct column names ---
            iata_code = row['iata']
            lat = row['latitude']
            lon = row['longitude']

            # Ensure data is valid before adding
            if pd.notna(iata_code) and pd.notna(lat) and pd.notna(lon):
                 # Convert IATA to string, strip whitespace, and make uppercase for consistent key lookup
                 iata_key = str(iata_code).strip().upper()
                 # Make sure the key isn't empty after stripping
                 if iata_key:
                    # Ensure lat/lon are valid floats before adding
                    try:
                        airport_data[iata_key] = (float(lat), float(lon))
                        processed_count += 1
                    except ValueError:
                        skipped_count += 1 # Count rows where lat/lon couldn't be converted
                 else:
                    skipped_count +=1 # Count rows with empty IATA after stripping
            else:
                skipped_count += 1 # Count rows with missing IATA, lat, or lon initially

        if skipped_count > 0:
             print(f"Skipped {skipped_count} rows in airport file due to missing or invalid iata/latitude/longitude.")

        if not airport_data:
            print(f"Warning: Airport coordinates file '{filepath}' loaded, but no valid data found or it was empty after processing. Orientation calculation skipped.")
            return None # Indicate failure due to empty/invalid data

        print(f"Successfully loaded coordinate data for {len(airport_data)} airports from '{filepath}'.")
        return airport_data

    except Exception as e:
        print(f"Error loading or processing airport coordinates file '{filepath}': {e}. Orientation calculation will be skipped.")
        return None # Indicate general loading failure

# --- Orientation Calculation ---
def calculate_orientation(origin, dest, airport_data):
    """Calculates flight orientation based on airport coordinates."""
    origin_str = str(origin).strip().upper() # Ensure uppercase lookup
    dest_str = str(dest).strip().upper()   # Ensure uppercase lookup

    if origin_str in airport_data and dest_str in airport_data:
        orig_lat, orig_lon = airport_data[origin_str]
        dest_lat, dest_lon = airport_data[dest_str]

        lat_diff = dest_lat - orig_lat
        lon_diff = dest_lon - orig_lon

        # Handle edge case where coordinates might be identical
        if abs(lat_diff) < 1e-6 and abs(lon_diff) < 1e-6:
             return 'Same Location'

        # Determine primary direction based on larger difference
        if abs(lat_diff) > abs(lon_diff):
            return 'North-South'
        else:
            return 'East-West'
    else:
        # Optional: Add debugging here if needed to see which specific airports are missing
        # missing = []
        # if origin_str not in airport_data: missing.append(f"Origin '{origin_str}'")
        # if dest_str not in airport_data: missing.append(f"Dest '{dest_str}'")
        # if missing: print(f"Debug: Orientation Unknown. Missing coords for: {', '.join(missing)}")
        return 'Unknown' # Return 'Unknown' if coords for either airport are missing

# --- Data Cleaning and Processing Function ---
def clean_and_process_flight_data(input_file, output_file, airport_coords_path):
    """
    Clean and process raw flight data.

    Args:
        input_file: Path to the raw flight data CSV
        output_file: Path to save the processed data CSV
        airport_coords_path: Path to the airport coordinates CSV file (iata-icao.csv)
    """
    print(f"\n--- Starting Cleaning for {input_file} ---")

    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"Read {len(df)} rows from raw data.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"Error reading input file {input_file}: {e}")
        return None

    # --- Load Airport Data ---
    airport_data = load_airport_data(airport_coords_path) # Returns None on failure

    # --- Column Selection ---
    relevant_columns = [
        'Year', 'Month', 'DayofMonth', 'DayOfWeek',
        'FlightDate', 'Reporting_Airline', 'Tail_Number',
        'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay',
        'CRSArrTime', 'ArrTime', 'ArrDelay', 'Cancelled', 'Diverted',
        'CRSElapsedTime', 'ActualElapsedTime', 'Distance'
    ]
    available_columns = [col for col in relevant_columns if col in df.columns]
    missing_columns = set(relevant_columns) - set(available_columns)
    if missing_columns:
        print(f"Info: The following potentially relevant columns are missing from the input: {missing_columns}")

    essential_for_processing = {'FlightDate', 'Tail_Number', 'Origin', 'Dest', 'CRSDepTime', 'CRSArrTime'}
    truly_missing = essential_for_processing - set(df.columns)
    if truly_missing:
        print(f"Error: Cannot proceed. Essential columns missing from input file: {truly_missing}")
        return None

    df = df[available_columns].copy() # Use .copy()

    # --- Basic Filtering (Cancelled/Diverted) ---
    initial_rows = len(df)
    if 'Cancelled' in df.columns:
        df['Cancelled'] = pd.to_numeric(df['Cancelled'], errors='coerce')
        df.dropna(subset=['Cancelled'], inplace=True)
        df = df[df['Cancelled'] == 0]
    if 'Diverted' in df.columns:
        df['Diverted'] = pd.to_numeric(df['Diverted'], errors='coerce')
        df.dropna(subset=['Diverted'], inplace=True)
        df = df[df['Diverted'] == 0]
    print(f"Removed {initial_rows - len(df)} cancelled/diverted/invalid status flights.")

    # --- Handle Missing Critical Data ---
    critical_columns = ['Tail_Number', 'Origin', 'Dest', 'FlightDate', 'CRSDepTime', 'CRSArrTime']
    critical_available = [col for col in critical_columns if col in df.columns]
    rows_before_na_drop = len(df)
    df.dropna(subset=critical_available, inplace=True)
    print(f"Removed {rows_before_na_drop - len(df)} rows with missing critical data ({critical_available}).")
    if df.empty: print("Error: No rows remaining after critical NA drop."); return None

    # --- Clean Identifiers (Tail Number, Origin, Dest) ---
    if 'Tail_Number' in df.columns:
        df['Tail_Number'] = df['Tail_Number'].astype(str).str.strip().str.upper()
        # Remove clearly invalid tail numbers
        invalid_tails = ['', '0', 'NONE', 'UNKN', 'UNKNOW', 'UNKNOWN']
        df = df[~df['Tail_Number'].isin(invalid_tails) & df['Tail_Number'].notna()]
    if 'Origin' in df.columns:
         df['Origin'] = df['Origin'].astype(str).str.strip().str.upper()
         df = df[df['Origin'].notna() & (df['Origin'] != '')]
    if 'Dest' in df.columns:
         df['Dest'] = df['Dest'].astype(str).str.strip().str.upper()
         df = df[df['Dest'].notna() & (df['Dest'] != '')]
    print(f"Cleaned Tail_Number, Origin, Dest. Rows remaining: {len(df)}")

    # --- Datetime Conversion ---
    try:
        # Attempt conversion, coerce errors to NaT (Not a Time)
        df['FlightDate'] = pd.to_datetime(df['FlightDate'], errors='coerce')
        rows_before_date_drop = len(df)
        df.dropna(subset=['FlightDate'], inplace=True) # Drop rows where conversion failed
        print(f"Removed {rows_before_date_drop - len(df)} rows with invalid FlightDate format.")
        if df.empty: print("Error: No valid FlightDate entries remaining."); return None
    except Exception as e: # Catch other potential errors during conversion
        print(f"Error converting 'FlightDate' to datetime: {e}")
        return None

    # --- Time Parsing (HHMM format) ---
    def parse_time(time_series):
        # Convert to string, remove potential '.0', pad with leading zeros to 4 digits
        time_str = time_series.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(4)
        # Replace '2400' (sometimes used for midnight) with '0000' for standard parsing
        time_str = time_str.replace('2400', '0000')
        # Parse as HHMM format, coerce errors to NaT
        parsed_time = pd.to_datetime(time_str, format='%H%M', errors='coerce').dt.time
        return parsed_time

    if 'CRSDepTime' in df.columns: df['CRSDepTime_parsed'] = parse_time(df['CRSDepTime'])
    if 'CRSArrTime' in df.columns: df['CRSArrTime_parsed'] = parse_time(df['CRSArrTime'])

    # Drop rows where time parsing failed for either departure or arrival
    time_cols_parsed = [col for col in ['CRSDepTime_parsed', 'CRSArrTime_parsed'] if col in df.columns]
    rows_before_time_drop = len(df)
    df.dropna(subset=time_cols_parsed, inplace=True)
    if rows_before_time_drop > len(df):
        print(f"Removed {rows_before_time_drop - len(df)} rows with invalid CRSDepTime/CRSArrTime formats.")
    if df.empty: print("Error: No rows remaining after cleaning time formats."); return None

    # --- Create Full Scheduled Datetime Objects ---
    if 'CRSDepTime_parsed' in df.columns:
        # Combine the date part of FlightDate with the parsed time part
        df['Schedule_DateTime'] = df.apply(
            lambda row: pd.Timestamp.combine(row['FlightDate'].date(), row['CRSDepTime_parsed']) if pd.notna(row['FlightDate']) and pd.notna(row['CRSDepTime_parsed']) else pd.NaT,
            axis=1
        )
    else:
        print("Error: CRSDepTime_parsed column missing. Cannot create Schedule_DateTime.")
        df['Schedule_DateTime'] = pd.NaT # Assign NaT explicitly


    if 'CRSArrTime_parsed' in df.columns and 'Schedule_DateTime' in df.columns:
         # Ensure we don't operate on NaT Schedule_DateTime
        valid_schedule_mask = df['Schedule_DateTime'].notna()
        df['Schedule_Arr_DateTime'] = pd.NaT # Initialize column with NaT
        # Calculate initial scheduled arrival datetime only for valid schedules
        df.loc[valid_schedule_mask, 'Schedule_Arr_DateTime'] = df[valid_schedule_mask].apply(
            lambda row: pd.Timestamp.combine(row['FlightDate'].date(), row['CRSArrTime_parsed']) if pd.notna(row['CRSArrTime_parsed']) else pd.NaT,
            axis=1
        )
        # Adjust arrival date for overnight flights ONLY where calculation was possible
        valid_arr_mask = df['Schedule_Arr_DateTime'].notna() & df['Schedule_DateTime'].notna()
        overnight_mask = valid_arr_mask & (df['Schedule_Arr_DateTime'] < df['Schedule_DateTime'])
        df.loc[overnight_mask, 'Schedule_Arr_DateTime'] += pd.Timedelta(days=1)
    else:
         print("Warning: Cannot accurately create Schedule_Arr_DateTime.")
         df['Schedule_Arr_DateTime'] = pd.NaT

    # Drop rows where Schedule_DateTime couldn't be created
    rows_before_sched_drop = len(df)
    df.dropna(subset=['Schedule_DateTime'], inplace=True)
    if rows_before_sched_drop > len(df):
        print(f"Removed {rows_before_sched_drop - len(df)} rows with invalid Schedule_DateTime.")
    if df.empty: print("Error: No rows remaining after Schedule_DateTime creation."); return None


    # --- Calculate Flight Duration ---
    if 'CRSElapsedTime' in df.columns:
        df['Flight_Duration_Minutes'] = pd.to_numeric(df['CRSElapsedTime'], errors='coerce')
        # Fallback calculation only if CRSElapsedTime resulted in NaN AND schedule times are valid
        nan_duration_mask = df['Flight_Duration_Minutes'].isna()
        can_calculate_mask = df['Schedule_Arr_DateTime'].notna() & df['Schedule_DateTime'].notna()
        calculate_mask = nan_duration_mask & can_calculate_mask

        if calculate_mask.any():
            print(f"Info: Calculating Flight_Duration_Minutes from schedule times for {calculate_mask.sum()} rows where CRSElapsedTime was missing/invalid.")
            df.loc[calculate_mask, 'Flight_Duration_Minutes'] = (df.loc[calculate_mask, 'Schedule_Arr_DateTime'] - df.loc[calculate_mask, 'Schedule_DateTime']).dt.total_seconds() / 60.0

    elif 'Schedule_Arr_DateTime' in df.columns and df['Schedule_Arr_DateTime'].notna().all() and \
         'Schedule_DateTime' in df.columns and df['Schedule_DateTime'].notna().all():
        # Calculate from scheduled datetimes if CRSElapsedTime column doesn't exist at all
        print("Info: Calculating Flight_Duration_Minutes from schedule times as CRSElapsedTime column is missing.")
        df['Flight_Duration_Minutes'] = (df['Schedule_Arr_DateTime'] - df['Schedule_DateTime']).dt.total_seconds() / 60.0
    else:
        print("Warning: Cannot determine Flight_Duration_Minutes reliably. Missing CRSElapsedTime and/or schedule times.")
        df['Flight_Duration_Minutes'] = np.nan # Assign NaN

    # --- Clean Flight Duration ---
    rows_before_duration_drop = len(df)
    # Drop rows where duration is still NaN
    df.dropna(subset=['Flight_Duration_Minutes'], inplace=True)
    # Drop rows with non-positive duration (likely data errors)
    df = df[df['Flight_Duration_Minutes'] > 0]
    if rows_before_duration_drop > len(df):
        print(f"Removed {rows_before_duration_drop - len(df)} rows with invalid/non-positive Flight_Duration_Minutes.")
    if df.empty: print("Error: No valid flights remaining after duration calculation."); return None

    # --- Calculate Flight Delay ---
    if 'ArrDelay' in df.columns:
        df['Flight_Delay'] = pd.to_numeric(df['ArrDelay'], errors='coerce')
        # If ArrDelay is missing/invalid, try filling with DepDelay
        if df['Flight_Delay'].isna().any() and 'DepDelay' in df.columns:
             print("Info: Filling missing ArrDelay values using DepDelay where possible.")
             df['Flight_Delay'].fillna(pd.to_numeric(df['DepDelay'], errors='coerce'), inplace=True)
    elif 'DepDelay' in df.columns:
        # Use DepDelay if ArrDelay column doesn't exist
        print("Info: Using DepDelay as ArrDelay column is missing.")
        df['Flight_Delay'] = pd.to_numeric(df['DepDelay'], errors='coerce')
    else:
        # If neither delay column exists, assume 0 delay
        df['Flight_Delay'] = 0
        print("Warning: Both ArrDelay and DepDelay are missing. Setting Flight_Delay to 0.")

    # Fill any remaining NaNs (e.g., if both ArrDelay/DepDelay were invalid) with 0
    df['Flight_Delay'].fillna(0, inplace=True)
    print("Processed Flight_Delay.")

    # --- Calculate Orientation ---
    if airport_data: # Only calculate if airport coordinate data was loaded successfully
        print("Calculating flight orientations...")
        df['Orientation'] = df.apply(lambda row: calculate_orientation(row['Origin'], row['Dest'], airport_data), axis=1)
        unknown_count = (df['Orientation'] == 'Unknown').sum()
        same_loc_count = (df['Orientation'] == 'Same Location').sum()
        print(f"Calculated Orientation. Unknown: {unknown_count}, Same Location: {same_loc_count}")
        if unknown_count > 0:
             print(f"Note: {unknown_count} flights have 'Unknown' orientation (likely missing airport coords in '{airport_coords_path}').")
    else:
        # If airport_data is None (failed to load/empty)
        print("Skipping orientation calculation due to missing/invalid airport coordinate data.")
        df['Orientation'] = 'Not Available' # Use a more descriptive placeholder


    # --- Final Touches ---
    # Rename columns for consistency
    rename_map = {'Reporting_Airline': 'Carrier_Airline'}
    df = df.rename(columns=rename_map)

    # Select final columns for output
    final_columns_base = [
        'Carrier_Airline', 'Tail_Number', 'Origin', 'Dest',
        'Schedule_DateTime', 'Flight_Duration_Minutes', 'Flight_Delay',
        'Orientation'
        # Add back other columns from relevant_columns if needed, e.g., 'Distance'
        # 'Distance'
    ]
    # Only include columns that actually exist in the dataframe at this point
    final_columns = [col for col in final_columns_base if col in df.columns]
    missing_final = set(final_columns_base) - set(final_columns)
    if missing_final:
        print(f"Warning: The following final columns could not be created/kept and will be missing from output: {missing_final}")

    # --- Save Processed Data ---
    try:
        df_output = df[final_columns].copy()
        # Ensure datetime format is consistent for saving
        df_output['Schedule_DateTime'] = pd.to_datetime(df_output['Schedule_DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_output.to_csv(output_file, index=False) # Use default date format saving or specify format='%Y-%m-%d %H:%M:%S'
        print(f"\n--- Cleaning Complete ---")
        print(f"Processed data saved to {output_file}")
        print(f"Total flights after processing: {len(df_output)}")
    except Exception as e:
        print(f"Error saving processed data to {output_file}: {e}")
        return None # Indicate failure

    # Return the dataframe with original datetime objects for chaining
    return df[final_columns]


# --- Flight Chain Creation Function ---
def create_flight_chains(df, output_dir, chain_length=3, max_time_diff_hours=24):
    """
    Create chains of consecutive flights for the same aircraft and split into train/val/test.

    Args:
        df: DataFrame containing processed flight data (output from clean_and_process_flight_data)
        output_dir: Directory to save the output files (train/val/test sequences)
        chain_length: Minimum number of flights required to form a sequence for analysis
        max_time_diff_hours: Maximum allowed time difference (in hours) between the
                             scheduled arrival of one flight and the scheduled departure
                             of the next flight for them to be considered consecutive.
    """
    if df is None or df.empty:
        print("Error: Input DataFrame for creating chains is empty or None. Aborting chain creation.")
        return None

    print(f"\n--- Starting Flight Chain Creation (Length={chain_length}, Max Gap={max_time_diff_hours}h) ---")

    if not os.path.exists(output_dir):
        print(f"Warning: Output directory {output_dir} does not exist. Attempting to create.")
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None

    # --- Input Validation ---
    required_chain_cols = {'Tail_Number', 'Schedule_DateTime', 'Flight_Duration_Minutes', 'Flight_Delay'}
    if not required_chain_cols.issubset(df.columns):
        print(f"Error: Missing required columns for chain creation: {required_chain_cols - set(df.columns)}")
        return None

    # --- Data Preparation ---
    try:
        # Ensure Schedule_DateTime is a datetime object (might be string if read from intermediate file)
        df['Schedule_DateTime'] = pd.to_datetime(df['Schedule_DateTime'], errors='coerce')
        df.dropna(subset=['Schedule_DateTime'], inplace=True) # Drop if conversion failed
        if df.empty:
            print("Error: No valid Schedule_DateTime found after coercion. Aborting chain creation.")
            return None
        # Ensure duration and delay are numeric
        df['Flight_Duration_Minutes'] = pd.to_numeric(df['Flight_Duration_Minutes'], errors='coerce')
        df['Flight_Delay'] = pd.to_numeric(df['Flight_Delay'], errors='coerce')
        df.dropna(subset=['Flight_Duration_Minutes', 'Flight_Delay'], inplace=True)

    except Exception as e:
        print(f"Error converting columns to correct types before chaining: {e}")
        return None

    # Sort flights by tail number and schedule time (essential for chaining)
    df = df.sort_values(['Tail_Number', 'Schedule_DateTime']).reset_index(drop=True)
    print(f"Sorted {len(df)} flights for chaining.")

    # --- Build Raw Chains ---
    flight_chains_raw = [] # Store chains as lists of dicts initially
    current_tail = None
    current_chain = []
    max_diff_timedelta = pd.Timedelta(hours=max_time_diff_hours)

    print("Iterating through flights to build chains...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building Chains"):
        row_dict = row.to_dict()
        # Skip row if critical chaining info is missing (should have been handled, but safety check)
        if pd.isna(row_dict['Schedule_DateTime']) or pd.isna(row_dict['Flight_Duration_Minutes']):
            continue

        current_schedule_dt = row_dict['Schedule_DateTime'] # Already datetime

        # Check if it's a new aircraft
        if row_dict['Tail_Number'] != current_tail:
            # If the previous chain was long enough, save it
            if len(current_chain) >= chain_length:
                flight_chains_raw.append(current_chain)

            # Start a new chain for the new tail number
            current_tail = row_dict['Tail_Number']
            current_chain = [row_dict]
        else:
            # Same aircraft, check time difference with the *previous* flight
            if not current_chain: # Safety check
                current_chain = [row_dict]
                continue

            prev_flight = current_chain[-1]
            # Safety check for NaT/NaN in previous flight before calculation
            if pd.isna(prev_flight['Schedule_DateTime']) or pd.isna(prev_flight['Flight_Duration_Minutes']):
                 if len(current_chain) >= chain_length: flight_chains_raw.append(current_chain)
                 current_chain = [row_dict] # Start new chain if prev flight was bad
                 continue

            # Calculate previous flight's scheduled arrival time
            prev_sched_arr_dt = prev_flight['Schedule_DateTime'] + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])

            # Calculate time difference (Current Dep - Prev Arr)
            time_diff = current_schedule_dt - prev_sched_arr_dt

            # Check if the time difference is within the allowed maximum gap
            # Require at least 1 minute ground time to be considered consecutive.
            if pd.Timedelta(minutes=1) <= time_diff <= max_diff_timedelta:
                # Add the current flight to the ongoing chain
                current_chain.append(row_dict)
            else:
                # Time difference too large or invalid (negative), break the chain
                if len(current_chain) >= chain_length:
                    flight_chains_raw.append(current_chain)

                # Start a new chain with the current flight
                current_chain = [row_dict]

    # Don't forget the very last chain being built
    if len(current_chain) >= chain_length:
        flight_chains_raw.append(current_chain)

    print(f"Identified {len(flight_chains_raw)} raw chains with at least {chain_length} flights.")
    if not flight_chains_raw:
        print("Warning: No flight chains could be created with the specified criteria. Check chain_length and max_time_diff_hours.")
        return None

    # --- Extract Features and Labels using Sliding Window ---
    features = []
    labels = [] # Regression label (actual delay)

    print(f"Extracting features from chains using sliding window (window size = {chain_length})...")
    chain_id_counter = 0
    skipped_sub_chains = 0
    for chain in tqdm(flight_chains_raw, desc="Processing Chains"):
        # Apply a sliding window of size 'chain_length' over each raw chain
        for i in range(len(chain) - chain_length + 1):
            sub_chain = chain[i : i + chain_length] # This is our analysis window

            # --- Validate sub_chain before processing ---
            valid_sub_chain = True
            for flight in sub_chain:
                 # Check for NaN in critical fields needed for feature generation
                 if pd.isna(flight['Schedule_DateTime']) or pd.isna(flight['Flight_Duration_Minutes']) or pd.isna(flight['Flight_Delay']):
                     valid_sub_chain = False
                     break
            if not valid_sub_chain:
                 skipped_sub_chains += 1
                 continue # Skip this sub_chain if any flight has critical NaNs

            # --- Feature Engineering for the window ---
            feature_dict = {
                'chain_id': chain_id_counter, # Unique ID for this specific sequence instance
                'tail_number': sub_chain[0]['Tail_Number'] # Tail number is constant within a chain
            }

            # Get all unique keys from all flights in the sub_chain to ensure consistent columns
            all_keys = set(key for flight in sub_chain for key in flight.keys())

            # Add features for each flight *within* the sub_chain window
            for j, flight in enumerate(sub_chain):
                flight_prefix = f'flight{j+1}_'
                for key in all_keys:
                    # Use .get() to handle cases where a key might be missing in one dict (though unlikely here)
                    feature_dict[flight_prefix + key] = flight.get(key, np.nan)

            # Calculate and add ground time between consecutive flights *within* the sub_chain
            ground_times_valid = True
            for j in range(1, chain_length): # Iterate from the second flight (index 1)
                prev_flight = sub_chain[j-1]
                curr_flight = sub_chain[j]

                # Already checked for NaNs at sub_chain level, but calculate safely
                prev_sched_arr_dt = prev_flight['Schedule_DateTime'] + pd.Timedelta(minutes=prev_flight['Flight_Duration_Minutes'])
                curr_sched_dep_dt = curr_flight['Schedule_DateTime']

                # Ground time in minutes
                ground_time = (curr_sched_dep_dt - prev_sched_arr_dt).total_seconds() / 60.0

                # Ground time should ideally be non-negative. Clamp at 0.
                feature_dict[f'ground_time_{j}'] = max(0, ground_time)
                if ground_time < 0: ground_times_valid = False # Flag if negative ground time occurred


            # --- Label Creation ---
            # The label is the delay of the *last* flight in the current window (sub_chain)
            last_flight_delay = sub_chain[-1]['Flight_Delay']
            labels.append(last_flight_delay)

            # --- Add Derived Features (PFD, FTD) ---
            # PFD: Previous Flight Delay
            # FTD: Flight Time Difference (essentially the calculated ground time)
            for k in range(1, chain_length): # k=1 => flight2 vs flight1; k=2 => flight3 vs flight2
                # PFD for flight k+1 is the delay of flight k (index k-1 in sub_chain)
                feature_dict[f'flight{k+1}_PFD'] = sub_chain[k-1]['Flight_Delay']
                # FTD for flight k+1 is the ground time *before* it (ground_time_k)
                feature_dict[f'flight{k+1}_FTD'] = feature_dict.get(f'ground_time_{k}', np.nan)

            features.append(feature_dict)
            chain_id_counter += 1

    if skipped_sub_chains > 0:
        print(f"Skipped {skipped_sub_chains} sub-chains due to internal NaN values.")

    if not features:
        print("Error: No feature sequences could be extracted from the chains.")
        return None

    # Convert features list of dicts to DataFrame
    feature_df = pd.DataFrame(features)

    # Add the labels (regression target)
    feature_df['delay_label'] = labels # Actual delay of the last flight in sequence

    # --- Create Classification Labels ---
    def categorize_delay(delay):
        if pd.isna(delay): return -1 # Assign -1 for NaN delays if any slip through
        if delay <= 0: return 0  # On time or early
        elif delay <= 15: return 1  # Slight delay (<= 15 min)
        elif delay <= 45: return 2  # Moderate delay (15 < delay <= 45 min)
        elif delay <= 120: return 3 # Significant delay (45 < delay <= 120 min)
        else: return 4  # Severe delay (> 120 min)

    feature_df['delay_category'] = feature_df['delay_label'].apply(categorize_delay)

    print(f"Generated {len(feature_df)} sequences of length {chain_length}.")
    print("Delay Category Distribution:")
    print(feature_df['delay_category'].value_counts(normalize=True).sort_index())


    # --- Split into Train/Validation/Test Sets ---
    np.random.seed(42) # for reproducibility
    n_samples = len(feature_df)
    if n_samples == 0:
        print("Error: No sequences generated, cannot split data.")
        return None

    indices = np.random.permutation(n_samples)

    train_split = int(0.7 * n_samples)
    val_split = int(0.15 * n_samples)
    # test_split is the remainder

    train_indices = indices[:train_split]
    val_indices = indices[train_split : train_split + val_split]
    test_indices = indices[train_split + val_split :]

    train_df = feature_df.iloc[train_indices].copy()
    val_df = feature_df.iloc[val_indices].copy()
    test_df = feature_df.iloc[test_indices].copy()

    # --- Save Processed Data ---
    full_output_path = os.path.join(output_dir, 'processed_flight_sequences_full.csv')
    train_output_path = os.path.join(output_dir, 'train_set.csv')
    val_output_path = os.path.join(output_dir, 'validation_set.csv')
    test_output_path = os.path.join(output_dir, 'test_set.csv')

    try:
        # Convert datetime columns to string format suitable for CSV if they exist
        for col in feature_df.columns:
             if 'Schedule_DateTime' in col:
                 feature_df[col] = pd.to_datetime(feature_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                 train_df[col] = pd.to_datetime(train_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                 val_df[col] = pd.to_datetime(val_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                 test_df[col] = pd.to_datetime(test_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')

        feature_df.to_csv(full_output_path, index=False)
        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        print(f"\n--- Chain Creation Complete ---")
        print(f"Saved full sequence data ({len(feature_df)}) to {full_output_path}")
        print(f"Train set ({len(train_df)} sequences) saved to {train_output_path}")
        print(f"Validation set ({len(val_df)} sequences) saved to {val_output_path}")
        print(f"Test set ({len(test_df)} sequences) saved to {test_output_path}")

    except Exception as e:
        print(f"Error saving sequence data files: {e}")
        return None

    # --- Save Last Event Per Tail (Optional but potentially useful) ---
    # Use the original cleaned df (before potential type conversion for saving sequences)
    try:
        # Find the index of the last flight for each tail number based on the sorted Schedule_DateTime
        last_event_indices = df.groupby('Tail_Number')['Schedule_DateTime'].idxmax()
        last_events_df = df.loc[last_event_indices]

        last_event_output_path = os.path.join(output_dir, 'last_event_per_tail.csv')
        last_events_df['Schedule_DateTime'] = pd.to_datetime(last_events_df['Schedule_DateTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        last_events_df.to_csv(last_event_output_path, index=False)
        print(f"Saved last known event for {len(last_events_df)} tail numbers to {last_event_output_path}")
    except Exception as e:
        print(f"Warning: Error saving last event data: {e}")
        # Continue even if this fails, main data is more important

    return feature_df # Return the full sequence dataframe (with datetimes as strings now)


# --- Main Execution Function ---
def main():
    # --- Determine Base Directory and Default Data Paths ---
    try:
        # Get the absolute path of the directory containing the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ is not defined, happens in interactive environments
        print("Warning: '__file__' not defined. Using current working directory as base.")
        print("Default data paths might be incorrect if script isn't run directly.")
        script_dir = os.getcwd()

    # Construct the path to the parent's parent's 'data' directory
    default_data_dir = os.path.join(script_dir, '..', '..', 'data')
    ml_data_dir = os.path.join(script_dir, '..', '..', 'mlData')
    # Normalize the path (e.g., resolves '..' components, handles separators)
    default_data_dir = os.path.normpath(default_data_dir)
    ml_data_dir = os.path.normpath(ml_data_dir)

    # Define default file paths based on the relative structure
    default_input_file = os.path.join(default_data_dir, DEFAULT_RAW_FLIGHT_FILENAME)
    default_coords_file = os.path.join(default_data_dir, DEFAULT_COORDS_FILENAME)
    default_output_dir = os.path.join(ml_data_dir, 'processedDataTest') # Default output relative to script

    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description='Clean raw flight data, create flight sequences for delay prediction using relative paths.')

    parser.add_argument('--input', type=str,
                        default=default_input_file,
                        help=f'Path to input raw flight data CSV (default: {default_input_file})')

    parser.add_argument('--coords-file', type=str,
                        default=default_coords_file,
                        help=f'Path to airport coordinates CSV (iata-icao.csv) (default: {default_coords_file})')

    parser.add_argument('--output-dir', type=str,
                        default=default_output_dir,
                        help=f'Directory to save processed data (cleaned file and sequences) (default: {default_output_dir})')

    parser.add_argument('--chain-length', type=int, default=3,
                        help='Number of consecutive flights in a sequence (default: 3)')

    # Renamed argument to match function parameter for clarity
    parser.add_argument('--max-time-diff-hours', type=int, default=24,
                        help='Maximum time gap in hours between consecutive flights in a chain (default: 24)')

    args = parser.parse_args()

    # --- Validate required input files ---
    if not os.path.exists(args.input):
        print(f"Error: Input flight data file not found at '{args.input}'")
        print("Please ensure the file exists or provide the correct path using --input.")
        exit(1) # Cannot proceed without input data

    if not os.path.exists(args.coords_file):
        print(f"Warning: Airport coordinates file not found at '{args.coords_file}'")
        print("Orientation calculation will be skipped or result in 'Unknown'/'Not Available'.")
        # Allow processing to continue

    # --- Execute Processing ---
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}")
            exit(1) # Cannot proceed if output directory cannot be created


    # Define the path for the intermediate cleaned file
    base_filename = os.path.splitext(os.path.basename(args.input))[0]
    cleaned_output_file = os.path.join(args.output_dir, f"{base_filename}_cleaned.csv")

    # --- Step 1: Clean the raw data ---
    cleaned_df = clean_and_process_flight_data(
        input_file=args.input,
        output_file=cleaned_output_file,
        airport_coords_path=args.coords_file # Pass the coordinates file path
    )

    # --- Step 2: Create flight chains and split data ---
    if cleaned_df is not None and not cleaned_df.empty:
        sequences_df = create_flight_chains(
            df=cleaned_df, # Use the DataFrame returned by the cleaning function
            output_dir=args.output_dir,
            chain_length=args.chain_length,
            max_time_diff_hours=args.max_time_diff_hours # Use the correctly named argument
        )
        if sequences_df is not None:
             print("\n--- Overall Process Completed Successfully ---")
        else:
             print("\n--- Process Completed with Errors during Chain Creation ---")
    elif cleaned_df is None:
         print("\n--- Process Stopped due to Errors during Data Cleaning ---")
    else: # cleaned_df is empty
         print("\n--- No valid flight data remained after cleaning. Cannot create chains. ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
