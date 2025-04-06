import os
import datetime
import numpy as np
import pandas as pd
import sys
sys.path.append('..')

from src.cleanData import (
    FlightDataProcessor,
    DATA_DIR,
    FILENAME_PATTERN,
    MONTH_SEASON_MAP,
    COLUMNS_TO_READ
)
# --- Configuration ---
# Input file from the previous script
PROCESSED_EVENT_FILE = "../processedData/processed_event_data.csv"
# Directory to save the final output
OUTPUT_DIR_CONGESTION = "../processedDataWithCongestion"
OUTPUT_FILENAME = "final_flight_data_with_congestion.csv"


class CongestionFeatureAdder:
    def __init__(self, processed_event_file: str):
        self.processed_event_file = processed_event_file
        self.flight_events_df = None

    def load_event_data(self) -> bool:
        """Loads the processed event data from the first script."""
        if not os.path.exists(self.processed_event_file):
            print(f"Error: Input file not found: {self.processed_event_file}")
            print("Please run the 'cleanRealTime.py' script first.")
            return False

        print(f"Loading processed event data from: {self.processed_event_file}")
        try:
            # Important: Parse Schedule_DateTime correctly when loading
            self.flight_events_df = pd.read_csv(
                self.processed_event_file,
                parse_dates=['Schedule_DateTime'] # Ensure datetime type
            )
            print(f"Loaded {len(self.flight_events_df)} records.")
            # Basic check for required columns
            required_cols = ['Schedule_DateTime', 'Origin', 'Dest', 'Orientation']
            if not all(col in self.flight_events_df.columns for col in required_cols):
                 missing = set(required_cols) - set(self.flight_events_df.columns)
                 print(f"Error: Loaded data is missing required columns: {missing}")
                 return False
            # Handle potential loading issues causing NaT
            if self.flight_events_df['Schedule_DateTime'].isnull().any():
                print("Warning: Found NaT values in Schedule_DateTime after loading. Dropping affected rows.")
                self.flight_events_df.dropna(subset=['Schedule_DateTime'], inplace=True)
                print(f"Rows remaining after dropping NaT: {len(self.flight_events_df)}")

            if self.flight_events_df.empty:
                print("Error: Dataframe is empty after loading or dropping NaT values.")
                return False

            return True

        except Exception as e:
            print(f"Error loading processed file {self.processed_event_file}: {e}")
            return False

    def calculate_airport_congestion_features(self, time_window='H') -> bool:
        """
        Calculates scheduled airport load features based on hourly counts.
        Adds 'Departures_In_Hour_Origin' and 'Arrivals_In_Hour_Dest' columns.
        """
        if self.flight_events_df is None or self.flight_events_df.empty:
            print("Error: Flight event data not loaded or empty.")
            return False

        print(f"\n--- Calculating Airport Congestion Features (Window: {time_window}) ---")
        df = self.flight_events_df.copy()

        # 1. Create the time window key
        try:
            df['Time_Window_Key'] = df['Schedule_DateTime'].dt.floor(time_window)
            print("Created time window key.")
        except Exception as e:
            print(f"Error creating time window key: {e}")
            return False

        # 2. Calculate scheduled departures per origin airport per time window
        print("Calculating departures per origin/time-window...")
        dep_counts = df[df['Orientation'] == 'Departure'].groupby(['Origin', 'Time_Window_Key']).size().reset_index(name='Departures_In_Window_Origin')

        # 3. Calculate scheduled arrivals per destination airport per time window
        print("Calculating arrivals per destination/time-window...")
        arr_counts = df[df['Orientation'] == 'Arrival'].groupby(['Dest', 'Time_Window_Key']).size().reset_index(name='Arrivals_In_Window_Dest')

        # 4. Merge departure counts back to the main dataframe
        print("Merging departure counts...")
        df = pd.merge(
            df,
            dep_counts,
            left_on=['Origin', 'Time_Window_Key'],
            right_on=['Origin', 'Time_Window_Key'],
            how='left'
        )
        # Fill NaN for flights where no other departures occurred in that window (should be 1 if only itself)
        df['Departures_In_Window_Origin'] = df['Departures_In_Window_Origin'].fillna(0).astype(int)


        # 5. Merge arrival counts back to the main dataframe
        print("Merging arrival counts...")
        df = pd.merge(
            df,
            arr_counts,
            left_on=['Dest', 'Time_Window_Key'],
            right_on=['Dest', 'Time_Window_Key'],
            how='left'
        )
        # Fill NaN for flights where no other arrivals occurred in that window (should be 1 if only itself)
        df['Arrivals_In_Window_Dest'] = df['Arrivals_In_Window_Dest'].fillna(0).astype(int)


        # Clean up the temporary key if desired
        # df.drop(columns=['Time_Window_Key'], inplace=True)

        self.flight_events_df = df
        print("Congestion features added:")
        print(f" - Departures_In_Window_Origin (Example: {df['Departures_In_Window_Origin'].iloc[:5].tolist()})")
        print(f" - Arrivals_In_Window_Dest (Example: {df['Arrivals_In_Window_Dest'].iloc[:5].tolist()})")
        print(f"Columns after adding congestion: {self.flight_events_df.columns.tolist()}")

        return True

    def save_final_data(self, output_dir: str = OUTPUT_DIR_CONGESTION, output_file: str = OUTPUT_FILENAME) -> None:
        """Saves the final dataframe with all features."""
        if self.flight_events_df is None or self.flight_events_df.empty:
            print("No final data to save.")
            return

        print(f"\n--- Saving Final Data ---")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        try:
            # Use a reasonable float format
            float_fmt = "%.4f"
            self.flight_events_df.to_csv(output_path, index=False, float_format=float_fmt)
            print(f"Final flight data saved with congestion features to {output_path}")
        except Exception as e:
            print(f"Error saving final flight data: {e}")

    def run(self) -> None:
        """Executes the full pipeline for adding congestion features."""
        print("Starting Congestion Feature Addition Pipeline...")
        start_time = datetime.datetime.now()

        if not self.load_event_data():
            print("Pipeline aborted due to loading errors.")
            return

        if not self.calculate_airport_congestion_features():
             print("Pipeline aborted due to errors in congestion calculation.")
             return

        self.save_final_data()

        end_time = datetime.datetime.now()
        print(f"\nCongestion Feature Addition Pipeline Complete. Total time: {end_time - start_time}")


if __name__ == "__main__":
    # Create an instance and run the pipeline
    pipeline = CongestionFeatureAdder(processed_event_file=PROCESSED_EVENT_FILE)
    pipeline.run()
