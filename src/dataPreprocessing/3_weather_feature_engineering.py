# src/dataPreprocessing/feature_engineering.py # Or weatherFeatureEngineering.py

import pandas as pd
import numpy as np
import pathlib
import sys
import os # For __file__ resolution
import traceback
from sklearn.preprocessing import OneHotEncoder # Example for encoding

def format_crs_dep_time(time_val):
    """Converts CRSDepTime (int/float HMM/HHMM) to HHMM string, handling NaN and 2400."""
    if pd.isna(time_val):
        return None # Or handle as appropriate, e.g., '0000' if default needed
    time_int = int(time_val)
    if time_int == 2400:
        return "0000" # Treat as midnight start of the day for joining
    else:
        return str(time_int).zfill(4)

def create_prediction_data():
    """
    Merges filtered flight data with weather data, performs basic feature
    engineering, and saves the result ready for modeling weather delays.
    """
    try:
        # --- 1. Setup Paths ---
        script_path = pathlib.Path(__file__).resolve()
        script_dir = script_path.parent
        project_root = script_dir.parent.parent
        data_dir = project_root / 'data'

        filtered_flights_file = data_dir / 'report_carrier_top10.csv'
        weather_file = data_dir / 'top10_weather_merged.csv'
        output_file = data_dir / 'weather_predict_data.csv' # output file name

        print(f"Project root directory: {project_root}")
        print(f"Data directory: {data_dir}")
        print(f"Input filtered flights file: {filtered_flights_file.name}")
        print(f"Input weather file: {weather_file.name}")
        print(f"Output file: {output_file.name}")

        # --- 2. Check Input File Existence ---
        if not filtered_flights_file.is_file():
            print(f"Error: Filtered flights data file not found at {filtered_flights_file}")
            print("Please run clean.py first.")
            sys.exit(1)
        if not weather_file.is_file():
            print(f"Error: Weather data file not found at {weather_file}")
            print("Please run merge.py first.")
            sys.exit(1)

        # --- 3. Load Data ---
        print("Loading filtered flight data...")
        try:
            flight_cols = [
                'Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate',
                'Reporting_Airline', 'Tail_Number', 'Flight_Number_Reporting_Airline',
                'Origin', 'OriginCityName', 'OriginState',
                'Dest', 'DestCityName', 'DestState',
                'CRSDepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15',
                'CRSArrTime', 'ArrDelay', 'ArrDelayMinutes', 'ArrDel15',
                'Cancelled', 'Diverted',
                'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Distance',
                'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay'
            ]
            flights_df = pd.read_csv(
                filtered_flights_file,
                usecols=flight_cols,
                parse_dates=['FlightDate'],
                low_memory=False
            )
            print(f"Filtered flight data loaded: {flights_df.shape}")
        except Exception as e:
            print(f"Error loading filtered flight data: {e}")
            traceback.print_exc()
            sys.exit(1)

        print("Loading weather data...")
        try:
            weather_cols = [
                'airport', 'timestamp', 'temp', 'humidity', 'wind_speed',
                'weather_main', 'cloudcover', 'precip_mm', 'pressure', 'visibility'
            ]
            weather_df = pd.read_csv(
                weather_file,
                usecols=weather_cols,
                parse_dates=['timestamp']
            )
            print(f"Weather data loaded: {weather_df.shape}")
        except Exception as e:
            print(f"Error loading weather data: {e}")
            traceback.print_exc()
            sys.exit(1)

        # --- 4. Prepare Data for Joining ---
        print("Preparing flight data for merge...")

        flights_df['CRSDepTime_Str'] = flights_df['CRSDepTime'].apply(format_crs_dep_time)
        valid_time_mask = flights_df['CRSDepTime_Str'].notna()
        flights_df_valid = flights_df[valid_time_mask].copy()
        print(f"Processing {flights_df_valid.shape[0]} flights with valid CRSDepTime format.")

        flights_df_valid['ScheduledDepDateTime_Str'] = flights_df_valid['FlightDate'].dt.strftime('%Y-%m-%d') + ' ' + \
                                                        flights_df_valid['CRSDepTime_Str'].str.slice(0, 2) + ':' + \
                                                        flights_df_valid['CRSDepTime_Str'].str.slice(2, 4) + ':00'
        flights_df_valid['ScheduledDepDateTime'] = pd.to_datetime(flights_df_valid['ScheduledDepDateTime_Str'], errors='coerce')

        rows_before_dropna = flights_df_valid.shape[0]
        flights_df_valid.dropna(subset=['ScheduledDepDateTime'], inplace=True)
        if rows_before_dropna > flights_df_valid.shape[0]:
             print(f"Dropped {rows_before_dropna - flights_df_valid.shape[0]} rows due to invalid datetime conversion during merge prep.")

        # ***** FIX 1: Address Deprecation Warning *****
        print("Calculating hourly departure timestamp using '.dt.floor(\'h\')'...")
        flights_df_valid['DepHourTimestamp'] = flights_df_valid['ScheduledDepDateTime'].dt.floor('h') # Changed 'H' to 'h'

        # ***** FIX 2: Correct column selection *****
        print("Selecting columns for flight data to merge...")
        # The list comprehension correctly includes 'Origin', 'DepHourTimestamp', and others
        # No need to add ['Origin', 'DepHourTimestamp'] explicitly at the start
        flight_merge_cols = [
            col for col in flights_df_valid.columns
            if col not in ['CRSDepTime_Str', 'ScheduledDepDateTime_Str'] # Exclude temporary helpers
        ]
        # Double-check that the required keys are indeed in the list (they should be)
        assert 'Origin' in flight_merge_cols, "Error: 'Origin' column missing before merge."
        assert 'DepHourTimestamp' in flight_merge_cols, "Error: 'DepHourTimestamp' column missing before merge."

        flights_to_merge = flights_df_valid[flight_merge_cols].copy() # Use .copy() for safety

        # Add a check here to be absolutely sure before merging
        if not flights_to_merge.columns.is_unique:
             print("Error: Columns in flights_to_merge are still not unique BEFORE merge!")
             print("Duplicate columns:", flights_to_merge.columns[flights_to_merge.columns.duplicated()].tolist())
             sys.exit(1)
        else:
             print("Columns in flights_to_merge are unique. Proceeding with merge.")


        print("Preparing weather data for merge...")
        weather_to_merge = weather_df.copy()

        # --- 5. Join (Merge) the Data ---
        print("Merging flight data with weather data...")
        merged_df = pd.merge(
            flights_to_merge,
            weather_to_merge,
            how='left',
            left_on=['Origin', 'DepHourTimestamp'],
            right_on=['airport', 'timestamp']
        )
        print(f"Merge complete. Resulting shape: {merged_df.shape}")

        # --- REST OF THE SCRIPT REMAINS THE SAME ---
        # --- 6. Post-Merge Cleanup & Verification ---
        print("Performing post-merge cleanup and verification...")
        weather_nan_check = merged_df[weather_cols[2:]].isnull().sum() # Check core weather features
        print("NaN counts in merged weather columns (should ideally be 0 after filtering):")
        print(weather_nan_check[weather_nan_check > 0])

        cols_to_drop = ['DepHourTimestamp', 'airport', 'timestamp']
        merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')
        print("Dropped redundant joining columns.")

        print("Cleaning target variable 'WeatherDelay' (filling NaN with 0)...")
        merged_df['WeatherDelay'] = merged_df['WeatherDelay'].fillna(0).astype(float)
        delay_cols = ['CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'DepDelayMinutes', 'ArrDelayMinutes']
        for col in delay_cols:
             if col in merged_df.columns:
                  merged_df[col] = merged_df[col].fillna(0).astype(float)

        # --- 7. Basic Feature Engineering ---
        print("Performing basic feature engineering...")

        merged_df['DepHour'] = merged_df['ScheduledDepDateTime'].dt.hour

        # Initialize dummy df variables to avoid NameError if columns don't exist
        weather_dummies = pd.DataFrame()
        airline_dummies = pd.DataFrame()

        if 'weather_main' in merged_df.columns:
            print("One-hot encoding 'weather_main'...")
            merged_df['weather_main'] = merged_df['weather_main'].fillna('Unknown')
            weather_dummies = pd.get_dummies(merged_df['weather_main'], prefix='WeatherIs', drop_first=True)
            merged_df = pd.concat([merged_df, weather_dummies], axis=1)
            merged_df = merged_df.drop(columns=['weather_main'])
            print(f"Added columns: {list(weather_dummies.columns)}")
        else:
            print("Warning: 'weather_main' column not found for encoding.")


        if 'Reporting_Airline' in merged_df.columns:
             print("One-hot encoding 'Reporting_Airline'...")
             airline_dummies = pd.get_dummies(merged_df['Reporting_Airline'], prefix='Airline', drop_first=True)
             merged_df = pd.concat([merged_df, airline_dummies], axis=1)
             # merged_df = merged_df.drop(columns=['Reporting_Airline']) # Decide whether to drop original
        else:
            print("Warning: 'Reporting_Airline' column not found for encoding.")


        numeric_weather_cols = ['temp', 'humidity', 'wind_speed', 'cloudcover', 'precip_mm', 'pressure', 'visibility']
        for col in numeric_weather_cols:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                if merged_df[col].isnull().any():
                    median_val = merged_df[col].median()
                    merged_df[col] = merged_df[col].fillna(median_val)
                    print(f"Filled NaNs in '{col}' with median value ({median_val}).")

        final_feature_cols = [
             'Month', 'DayOfWeek', 'DepHour',
             'CRSElapsedTime', 'Distance',
             'temp', 'humidity', 'wind_speed', 'cloudcover', 'precip_mm', 'pressure', 'visibility',
             'WeatherDelay' # Target Variable
        ]
        # Add the dynamically generated dummy columns IF they were created
        if not weather_dummies.empty:
            final_feature_cols.extend(list(weather_dummies.columns))
        if not airline_dummies.empty:
            final_feature_cols.extend(list(airline_dummies.columns))

        final_columns_present = [col for col in final_feature_cols if col in merged_df.columns]
        if 'WeatherDelay' not in final_columns_present and 'WeatherDelay' in merged_df.columns:
             final_columns_present.append('WeatherDelay') # Ensure target is there

        ids_to_keep = ['FlightDate', 'Reporting_Airline', 'Origin', 'Dest', 'Tail_Number', 'Flight_Number_Reporting_Airline', 'ScheduledDepDateTime']
        final_columns_with_ids = ids_to_keep + final_columns_present
        final_columns_with_ids = [col for col in final_columns_with_ids if col in merged_df.columns]

        final_columns_set = set()
        final_columns_unique = []
        for col in final_columns_with_ids:
            if col not in final_columns_set:
                final_columns_set.add(col)
                final_columns_unique.append(col)

        final_df = merged_df[final_columns_unique].copy()
        print(f"Selected final columns. Resulting shape: {final_df.shape}")
        print(f"Final columns: {final_df.columns.tolist()}")


        # --- 8. Save the Final Dataset ---
        print(f"Saving final processed data to: {output_file}")
        try:
            final_df.to_csv(output_file, index=False)
            print("Successfully saved the final dataset.")
        except Exception as e:
            print(f"Error saving final data: {e}")
            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the feature engineering process: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- Run the function when the script is executed ---
if __name__ == "__main__":
    create_prediction_data()
