import os
import datetime
import numpy  as np
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

# Set PROCESSED_FILE to a specific file path for preprocessed data.
# If left empty, the full pipeline will run from raw data.
PROCESSED_FILE = "..\\processedDataRealtime\\testP_set.csv"

class FullPipelineMerger:
    def __init__(self,
                 processed_file : str = PROCESSED_FILE):
        self.processed_file    = processed_file
        self.processed_flights = None

    def load_data(self) -> bool:
        # If a processed file is specified, load that data directly.
        if self.processed_file:
            print(f"Using specified processed file: {self.processed_file}")
            try:
                self.processed_flights = pd.read_csv(self.processed_file)
                print(f"Loaded {len(self.processed_flights)} records from {self.processed_file}.")
            except Exception as e:
                print(f"Error loading processed file {self.processed_file}: {e}")
                return False
        else:
            # Otherwise, run the full pipeline from raw data.
            processor = FlightDataProcessor(
                data_dir        = DATA_DIR,
                file_pattern    = FILENAME_PATTERN,
                month_map       = MONTH_SEASON_MAP,
                columns_to_read = COLUMNS_TO_READ
            )
            if not processor.load_and_prepare_initial_data():
                print("Data loading failed.")
                return False
            if not processor.preprocess_data():
                print("Preprocessing failed.")
                return False
            if not processor.reshape_and_calculate_features():
                print("Reshaping and feature calculation failed.")
                return False

            # Call the split function as part of the pipeline.
            train_df, val_df, test_df = processor.split_and_sample_data()
            print(f"Train, Validation, and Test sets created. (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")

            self.processed_flights    = processor.final_flight_df
            if self.processed_flights is None or self.processed_flights.empty:
                print("No processed flight data available.")
                return False
        return True

    def perform_queue_analysis(self) -> pd.DataFrame:
        # For each Tail_Number, calculate average FTD and Flight Duration.
        analysis_results = []
        for tail, group in self.processed_flights.groupby("Tail_Number"):
            avg_ftd      = group["FTD"].mean()
            avg_duration = group["Flight_Duration_Minutes"].mean() if "Flight_Duration_Minutes" in group.columns else np.nan
            lam          = 1 / avg_ftd      if avg_ftd           and avg_ftd      > 0 else np.nan
            mu           = 1 / avg_duration if avg_duration      and avg_duration > 0 else np.nan
            stable       = lam < mu         if not np.isnan(lam) and not np.isnan(mu) else False

            analysis_results.append({
                "Tail_Number"             : tail,
                "Avg_FTD_min"             : avg_ftd,
                "Avg_Flight_Duration_min" : avg_duration,
                "Arrival_Rate_lambda"     : lam,
                "Service_Rate_mu"         : mu,
                "Stable_Queue"            : stable
            })

        queue_summary = pd.DataFrame(analysis_results)
        print("Queue analysis summary created.")
        return queue_summary

    def merge_queue_features(self, queue_summary: pd.DataFrame) -> pd.DataFrame:
        updated_flights = pd.merge(self.processed_flights, queue_summary, on="Tail_Number", how="left")
        return updated_flights

    def save_updated_data(self, updated_flights: pd.DataFrame,
                          output_dir   : str = "../processedDataQUpdated",
                          output_file  : str = "processed_flights_with_queue_features.csv") -> None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        try:
            updated_flights.to_csv(output_path, index=False, float_format="%.4f")
            print(f"Updated flight data saved with queue features to {output_path}")
        except Exception as e:
            print(f"Error saving updated flight data: {e}")

    def run(self) -> None:
        print("Starting full processing pipeline with split and queue analysis merge...")
        start_time = datetime.datetime.now()

        if not self.load_data():
            return

        queue_summary   = self.perform_queue_analysis()
        updated_flights = self.merge_queue_features(queue_summary)
        self.save_updated_data(updated_flights)

        end_time = datetime.datetime.now()
        print(f"Full pipeline and merge update complete. Total time: {end_time - start_time}")

if __name__ == "__main__":
    pipeline = FullPipelineMerger(processed_file=PROCESSED_FILE)
    pipeline.run()
