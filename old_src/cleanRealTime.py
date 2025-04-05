import datetime
import os
import pandas      as pd
import numpy       as np
import types

from cleanData   import (
    FlightDataProcessor, 
    DATA_DIR, 
    FILENAME_PATTERN, 
    MONTH_SEASON_MAP, 
    COLUMNS_TO_READ, 
    _parse_datetime
)

# Set FILENAME to a specific file path
# For example:
#   Unix:    FILENAME = "../data/Data.csv"
#   Windows: FILENAME = "..\\data\\Data.csv"
FILENAME   = "..\\data\\On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_0.csv"
OUTPUT_DIR = '../processedDataRealtime'

class RealtimeFlightPipeline:
    def __init__(self,
                 filename   : str = None,
                 output_dir : str = OUTPUT_DIR):
        self.filename   = filename
        self.output_dir = output_dir
        self.processor  = None

    @staticmethod
    def reshape_and_calculate_features_realtime(self):
        """
        Real-time version of reshape_and_calculate_features:
         - Parses datetime columns.
         - Drops rows with missing departure times but retains rows with missing arrival times.
         - Computes flight duration only when arrival times are available.
         - Proceeds to build the departure and arrival frames and calculate FTD/PFD.
        """
        print("\n--- Stage 3: Reshaping Data and Calculating Features (Real-time Mode) ---")
        df = self.processed_df.copy()

        print("Parsing schedule departure/arrival times...")
        df['Schedule_Departure_DT'] = _parse_datetime(df, 'FlightDate', 'CRSDepTime')
        df['Schedule_Arrival_DT']   = _parse_datetime(df, 'FlightDate', 'CRSArrTime')
    
        # Only drop rows where departure datetime is invalid
        rows_before   = len(df)
        df.dropna(subset=['Schedule_Departure_DT'], inplace=True)
        removed_count = rows_before - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with invalid schedule departure datetimes.")

        # Adjust overnight arrivals only if arrival time is present
        valid_arrival       = df['Schedule_Arrival_DT'].notna()
        overnight_condition = valid_arrival & (df['Schedule_Arrival_DT'] <= df['Schedule_Departure_DT'])
        num_overnight       = overnight_condition.sum()
        print(f" -> Identified {num_overnight} potential overnight arrivals requiring date adjustment.")
        if num_overnight > 0:
            df.loc[overnight_condition, 'Schedule_Arrival_DT'] = df.loc[overnight_condition, 'Schedule_Arrival_DT'] + pd.Timedelta(days=1)
            print(" -> Arrival dates adjusted.")

        print("Calculating flight duration from corrected schedule times...")
        # Calculate duration only for rows with both departure and arrival times.
        valid_times = df['Schedule_Departure_DT'].notna() & df['Schedule_Arrival_DT'].notna()
        df['Calculated_Duration_Minutes']                  = np.nan
        df.loc[valid_times, 'Calculated_Duration_Minutes'] = (
            (df.loc[valid_times, 'Schedule_Arrival_DT'] - 
             df.loc[valid_times, 'Schedule_Departure_DT']).dt.total_seconds() / 60.0
        )
        print(f"Rows after datetime processing & duration calculation: {len(df)}")
        if len(df) == 0:
            print("Error: No valid data remaining after datetime processing.")
            return False

        # --- Create Departure and Arrival Frames ---
        print("Creating departure and arrival frames...")
        common_cols_base = [
            'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Tail_Number',
            'Origin', 'Dest', 'Season', 'Calculated_Duration_Minutes'
        ]
        common_cols    = [col for col in common_cols_base if col in df.columns]
        missing_common = set(common_cols_base) - set(common_cols)
        if missing_common:
            print(f"Warning: Columns missing for reshape: {missing_common}. They won't be in the final dataset.")

        cols_for_dep = common_cols + ['Schedule_Departure_DT', 'DepDelayMinutes']
        cols_for_arr = common_cols + ['Schedule_Arrival_DT',   'ArrDelayMinutes']

        dep_df = df[[col for col in cols_for_dep if col in df.columns]].copy()
        dep_df['Orientation'] = 'Departure'
        dep_df.rename(columns={
            'Schedule_Departure_DT'           : 'Schedule_DateTime',
            'DepDelayMinutes'                 : 'Flight_Delay',
            'Reporting_Airline'               : 'Carrier_Airline',
            'Flight_Number_Reporting_Airline' : 'Flight_Number',
            'Calculated_Duration_Minutes'     : 'Flight_Duration_Minutes'
        }, inplace=True)

        arr_df = df[[col for col in cols_for_arr if col in df.columns]].copy()
        arr_df['Orientation'] = 'Arrival'
        arr_df.rename(columns={
            'Schedule_Arrival_DT'             : 'Schedule_DateTime',
            'ArrDelayMinutes'                 : 'Flight_Delay',
            'Reporting_Airline'               : 'Carrier_Airline',
            'Flight_Number_Reporting_Airline' : 'Flight_Number',
            'Calculated_Duration_Minutes'     : 'Flight_Duration_Minutes'
        }, inplace=True)

        print("Concatenating frames...")
        flight_df                 = pd.concat([dep_df, arr_df],
                                              ignore_index=True)
        flight_df['Flight_Delay'] = pd.to_numeric(flight_df['Flight_Delay'],
                                                  errors='coerce').fillna(0)
        if 'Flight_Duration_Minutes' in flight_df.columns:
            flight_df['Flight_Duration_Minutes'] = pd.to_numeric(flight_df['Flight_Duration_Minutes'], 
                                                                 errors='coerce').fillna(0)

        print(f"Reshaped data frame ('Flight') created with {len(flight_df)} rows.")
        print(f"Columns in reshaped frame: {flight_df.columns.tolist()}")
        if flight_df.empty:
            print("Error: Reshaped DataFrame is empty. Cannot calculate FTD/PFD.")
            return False

        # --- FTD and PFD Calculation ---
        flight_df['FTD'] = 0.0  # Time since last event for this tail
        flight_df['PFD'] = 0.0  # Previous Flight Delay

        print("Calculating Flight Time Duration (FTD)...")
        flight_df.sort_values(by=['Tail_Number', 'Schedule_DateTime'], inplace=True)
        flight_df.reset_index(drop=True, inplace=True)
        flight_df['FTD_Timedelta'] = flight_df.groupby('Tail_Number')['Schedule_DateTime'].diff()
        flight_df['FTD']           = flight_df['FTD_Timedelta'].dt.total_seconds() / 60.0
        flight_df['FTD']           = flight_df['FTD'].fillna(0.0)
        flight_df.drop(columns=['FTD_Timedelta'], inplace=True)

        print("Calculating Previous Flight Delay (PFD)...")
        flight_df['PFD'] = flight_df.groupby('Tail_Number')['Flight_Delay'].shift(1)
        flight_df['PFD'] = flight_df['PFD'].fillna(0.0)

        print("Identifying last recorded event for each tail number...")
        is_last            = (flight_df['Tail_Number'] != flight_df['Tail_Number'].shift(-1)) | (flight_df.index == len(flight_df) - 1)
        self.last_point_df = flight_df[is_last].copy()
        print(f"Found {len(self.last_point_df)} last points (last event per tail).")

        self.final_flight_df = flight_df
        print("\n--- Stage 3 Complete ---")
        return True

    def run(self):
        print("Starting realtime flight data processing...")
        start_time = datetime.datetime.now()

        # Create FlightDataProcessor instance
        self.processor = FlightDataProcessor(
            data_dir        = DATA_DIR,
            file_pattern    = FILENAME_PATTERN,
            month_map       = MONTH_SEASON_MAP,
            columns_to_read = COLUMNS_TO_READ
        )

        # Quick test mode: load the specified file if FILENAME is provided
        if self.filename:
            print(f"Quick test mode: Loading specified file: {self.filename}")
            try:
                df = pd.read_csv(self.filename, usecols=COLUMNS_TO_READ, low_memory=False)
                # For a single file, assign a default Season value
                df['Season']          = 'RealTime'
                self.processor.raw_df = df
                print(f"Loaded {len(df)} records from {self.filename}.")
            except Exception as e:
                print(f"Error loading specified file {self.filename}: {e}")
                return
        else:
            if not self.processor.load_and_prepare_initial_data():
                print("Data loading failed.")
                return

        if not self.processor.preprocess_data():
            print("Preprocessing failed.")
            return

        # Monkey-patch the reshape function with our real-time version that preserves empty arrivals
        self.processor.reshape_and_calculate_features = types.MethodType(RealtimeFlightPipeline.reshape_and_calculate_features_realtime, self.processor)
        if not self.processor.reshape_and_calculate_features():
            print("Reshaping and feature calculation failed.")
            return

        # For real-time prediction, use the final processed DataFrame without splitting.
        final_df = self.processor.final_flight_df
        if final_df is None or final_df.empty:
            print("No processed data available for saving.")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        test_set_path = os.path.join(self.output_dir, "testP_set.csv")
        try:
            final_df.to_csv(test_set_path, index=False, float_format='%.2f')
            print(f"Saved realtime test set to {test_set_path}")
        except Exception as e:
            print(f"Error saving realtime test set: {e}")

        end_time = datetime.datetime.now()
        print(f"Realtime processing complete. Total time: {end_time - start_time}")

if __name__ == "__main__":
    pipeline = RealtimeFlightPipeline(filename=FILENAME)
    pipeline.run()