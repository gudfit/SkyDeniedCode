# src/dataPreprocessing/clean.py

import pandas as pd
import pathlib
import sys
import os # For __file__ resolution
import traceback

def clean_flight_data_based_on_weather():
    """
    Filters the merged flight data ('ReportingCarrierMerged.csv') to keep only
    flights whose origin airport and scheduled departure hour match an entry
    in the merged weather data ('Top10WeatherMerged.csv'). Saves the result
    to 'ReportCarrierTop10.csv'.
    """
    try:
        # --- 1. Setup Paths ---
        script_path = pathlib.Path(__file__).resolve()
        script_dir = script_path.parent
        project_root = script_dir.parent.parent
        data_dir = project_root / 'data'

        weather_file = data_dir / 'top10_weather_merged.csv'
        flights_file = data_dir / 'report_carrier_merged.csv'
        output_file = data_dir / 'report_carrier_top10.csv'

        print(f"Project root directory: {project_root}")
        print(f"Data directory: {data_dir}")
        print(f"Input weather file: {weather_file.name}")
        print(f"Input flights file: {flights_file.name}")
        print(f"Output file: {output_file.name}")

        # --- 2. Check Input File Existence ---
        if not weather_file.is_file():
            print(f"Error: Weather data file not found at {weather_file}")
            sys.exit(1)
        if not flights_file.is_file():
            print(f"Error: Flights data file not found at {flights_file}")
            sys.exit(1)

        # --- 3. Load Data ---
        print("Loading weather data...")
        try:
            weather_df = pd.read_csv(weather_file, parse_dates=['timestamp'])
            print(f"Weather data loaded: {weather_df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading weather data: {e}")
            sys.exit(1)

        print("Loading flight data...")
        try:
            # Use low_memory=False for large files to prevent DtypeWarning
            flights_df = pd.read_csv(flights_file, low_memory=False)
            print(f"Flight data loaded: {flights_df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading flight data: {e}")
            sys.exit(1)

        # --- 4. Prepare Weather Keys ---
        print("Processing weather data to create filter keys...")
        # Ensure timestamp is datetime
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        # Create a set of (airport_code, timestamp_hour) tuples for efficient lookup
        # The weather timestamp is already hourly
        weather_keys = set(zip(weather_df['airport'], weather_df['timestamp']))
        print(f"Created {len(weather_keys)} unique weather keys (Airport, HourTimestamp).")
        if not weather_keys:
             print("Warning: No weather keys were generated. The output file will likely be empty.")


        # --- 5. Prepare Flight Data and Keys ---
        print("Processing flight data...")

        # Convert FlightDate to datetime (only date part needed initially)
        flights_df['FlightDate'] = pd.to_datetime(flights_df['FlightDate']).dt.date

        # Format CRSDepTime (scheduled departure time) HMM or HHMM -> HHMM string
        # Handle potential NaNs or non-numeric types before conversion
        flights_df['CRSDepTime'] = pd.to_numeric(flights_df['CRSDepTime'], errors='coerce').fillna(0).astype(int)

        # Convert CRSDepTime to HHMM string, padding with leading zeros
        # Handle 2400 -> 0000 (representing midnight at the start of the *next* day conceptually,
        # but for matching weather *on the hour*, we align it with 00:00 of the *current* flight date)
        # Simpler approach first: treat as '0000' of the same day.
        def format_time(time_int):
            if time_int == 2400:
                return "0000"
            else:
                return str(time_int).zfill(4)

        flights_df['CRSDepTime_Str'] = flights_df['CRSDepTime'].apply(format_time)

        # Ensure the formatted time string is valid (HHMM)
        # This regex checks for HH between 00-23 and MM between 00-59
        valid_time_mask = flights_df['CRSDepTime_Str'].str.match(r'^([01]\d|2[0-3])([0-5]\d)$')
        original_rows = flights_df.shape[0]
        flights_df = flights_df[valid_time_mask].copy() # Filter out invalid times
        print(f"Filtered out {original_rows - flights_df.shape[0]} rows with invalid CRSDepTime format.")


        # Combine Date and Time String
        # Ensure FlightDate is string before concatenation
        flights_df['ScheduledDepDateTime_Str'] = flights_df['FlightDate'].astype(str) + ' ' + \
                                                 flights_df['CRSDepTime_Str'].str.slice(0, 2) + ':' + \
                                                 flights_df['CRSDepTime_Str'].str.slice(2, 4) + ':00'

        # Convert to datetime objects
        print("Converting combined date/time strings to datetime objects...")
        flights_df['ScheduledDepDateTime'] = pd.to_datetime(flights_df['ScheduledDepDateTime_Str'], errors='coerce')

        # Drop rows where conversion failed (should be rare after filtering, but good practice)
        rows_before_dropna = flights_df.shape[0]
        flights_df.dropna(subset=['ScheduledDepDateTime'], inplace=True)
        if rows_before_dropna > flights_df.shape[0]:
             print(f"Dropped {rows_before_dropna - flights_df.shape[0]} rows due to invalid datetime conversion.")

        # Create the hourly timestamp key by rounding *down* to the nearest hour
        print("Rounding scheduled departure time down to the hour...")
        flights_df['DepHourTimestamp'] = flights_df['ScheduledDepDateTime'].dt.floor('H')

        # Create the composite key (Origin Airport, Hourly Timestamp) for filtering
        print("Creating filter key (Origin, DepHourTimestamp) for flights...")
        flights_df['FilterKey'] = list(zip(flights_df['Origin'], flights_df['DepHourTimestamp']))

        # --- 6. Filter Flight Data ---
        print(f"Filtering {flights_df.shape[0]} flight records based on weather keys...")
        # Use the pre-computed set of weather keys for efficient filtering with `isin`
        filtered_flights_df = flights_df[flights_df['FilterKey'].isin(weather_keys)].copy()
        print(f"Filtering complete. Kept {filtered_flights_df.shape[0]} flight records.")

        # --- 7. Clean up and Save ---
        # Drop the temporary helper columns before saving
        columns_to_drop = ['CRSDepTime_Str', 'ScheduledDepDateTime_Str', 'ScheduledDepDateTime', 'DepHourTimestamp', 'FilterKey']
        # Only drop columns that actually exist to avoid errors if script is rerun/modified
        columns_to_drop = [col for col in columns_to_drop if col in filtered_flights_df.columns]
        if columns_to_drop:
             filtered_flights_df = filtered_flights_df.drop(columns=columns_to_drop)
             print(f"Dropped temporary helper columns: {', '.join(columns_to_drop)}")

        print(f"Saving filtered flight data to: {output_file}")
        try:
            filtered_flights_df.to_csv(output_file, index=False)
            print("Successfully saved filtered data.")
        except Exception as e:
            print(f"Error saving filtered data: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the cleaning process: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- Run the function when the script is executed ---
if __name__ == "__main__":
    clean_flight_data_based_on_weather()
