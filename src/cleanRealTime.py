# run_realtime.py
import datetime
import os
import pandas as pd
from cleanData import FlightDataProcessor, DATA_DIR, FILENAME_PATTERN, MONTH_SEASON_MAP, COLUMNS_TO_READ

# Set FILENAME to a specific file path
# For example
# - Unix:    FILENAME = "../data/Data.csv"
# - Windows: FILENAME = "..\\data\\Data.csv"
FILENAME = "..\\data\\On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_0.csv"

# Configure the output directory for realtime predictions
OUTPUT_DIR = '../processedDataRealtime'

def run_realtime_processing():
    print("Starting realtime flight data processing...")
    start_time = datetime.datetime.now()

    processor = FlightDataProcessor(
        data_dir        = DATA_DIR,
        file_pattern    = FILENAME_PATTERN,
        month_map       = MONTH_SEASON_MAP,
        columns_to_read = COLUMNS_TO_READ
    )

    if FILENAME:
        print(f"Quick test mode: Loading specified file: {FILENAME}")
        try:
            df = pd.read_csv(FILENAME, usecols=COLUMNS_TO_READ, low_memory=False)
            # If testing a single file, assign a default Season value
            df['Season'] = 'Test'
            processor.raw_df = df
            print(f"Loaded {len(df)} records from {FILENAME}.")
        except Exception as e:
            print(f"Error loading specified file {FILENAME}: {e}")
            return
    else:
        if not processor.load_and_prepare_initial_data():
            print("Data loading failed.")
            return

    if not processor.preprocess_data():
        print("Preprocessing failed.")
        return

    if not processor.reshape_and_calculate_features():
        print("Reshaping and feature calculation failed.")
        return

    # For real-time prediction, we use the final processed DataFrame without splitting.
    final_df = processor.final_flight_df
    if final_df is None or final_df.empty:
        print("No processed data available for saving.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_set_path = os.path.join(OUTPUT_DIR, "testP_set.csv")
    try:
        final_df.to_csv(test_set_path, index=False, float_format='%.2f')
        print(f"Saved realtime test set to {test_set_path}")
    except Exception as e:
        print(f"Error saving realtime test set: {e}")

    end_time = datetime.datetime.now()
    print(f"Realtime processing complete. Total time: {end_time - start_time}")

if __name__ == "__main__":
    run_realtime_processing()
