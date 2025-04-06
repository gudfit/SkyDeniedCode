# src/dataPreprocessing/merge.py

import pandas as pd
import pathlib
import sys
import os # os is needed indirectly via __file__ resolution on some systems
import traceback

def merge_files(data_dir: pathlib.Path, file_pattern: str, output_filename: str, expected_files_list: list = None):
    """
    Merges CSV files matching a given pattern within a specified directory
    into a single output CSV file.

    Args:
        data_dir: The Path object representing the directory containing the input CSVs.
        file_pattern: The glob pattern to match the input CSV files.
        output_filename: The name for the merged output CSV file.
        expected_files_list: An optional list of specific filenames expected to be found.
                             Used for more informative error messages.
    """
    output_file_path = data_dir / output_filename
    print(f"\n--- Starting merge process for '{output_filename}' ---")
    print(f"Data directory: {data_dir}")
    print(f"Looking for files matching: {file_pattern}")

    # --- Find the input files ---
    input_files = sorted(list(data_dir.glob(file_pattern))) # Sort for predictable order

    if not input_files:
        print(f"Error: No files found matching the pattern '{file_pattern}' in {data_dir}")
        if expected_files_list:
            print("Please ensure the following files exist in the 'data' directory:")
            for fname in expected_files_list:
                print(f" - {fname}")
        # Since this function might be called multiple times, return False on failure
        # instead of exiting the whole script immediately.
        return False

    print(f"Found {len(input_files)} files to merge:")
    for f in input_files:
        print(f" - {f.name}")

    # --- Read and concatenate the files ---
    df_list = []
    read_error = False
    for file_path in input_files:
        print(f"Reading {file_path.name}...")
        try:
            # Read CSV, handling potential type inconsistencies if needed
            df = pd.read_csv(file_path, low_memory=False)
            # Check if the last column is unnamed and empty, often added by Excel/exports
            if df.columns[-1].startswith('Unnamed:'):
                # Check if the column is truly empty (all NaN) before dropping
                if df[df.columns[-1]].isnull().all():
                    print(f"  -> Dropping empty last column '{df.columns[-1]}' from {file_path.name}")
                    df = df.iloc[:, :-1]
                else:
                    print(f"  -> Warning: Last column '{df.columns[-1]}' starts with 'Unnamed:' but contains data. Not dropping.")

            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            # Mark error and continue to see if other files can be read,
            # but prevent merging incomplete data later.
            read_error = True
            # Optional: break here if one file error should stop this specific merge
            # break

    if read_error:
         print(f"Errors occurred while reading some files for '{output_filename}'. Aborting this merge.")
         return False # Indicate failure for this merge task

    if not df_list:
        print(f"Error: No dataframes were successfully read for '{output_filename}'. Cannot merge.")
        return False # Indicate failure

    print("Concatenating dataframes...")
    merged_df = pd.concat(df_list, ignore_index=True)

    # --- Save the merged file ---
    print(f"Saving merged data to: {output_file_path}")
    try:
        merged_df.to_csv(output_file_path, index=False)
        print(f"Successfully merged files and saved '{output_filename}'.")
        print(f"Merged DataFrame shape: {merged_df.shape}")
        return True # Indicate success for this merge task
    except Exception as e:
        print(f"Error saving merged file to {output_file_path}: {e}")
        return False # Indicate failure


def run_all_merges():
    """
    Finds and merges multiple sets of data files.
    """
    overall_success = True
    try:
        # --- Determine paths relative to this script ---
        script_path = pathlib.Path(__file__).resolve()
        script_dir = script_path.parent
        project_root = script_dir.parent.parent
        data_dir = project_root / 'data'

        print(f"Project root directory: {project_root}")
        print(f"Located data directory: {data_dir}")

        if not data_dir.is_dir():
            print(f"Error: Data directory not found at {data_dir}")
            sys.exit(1)

        # --- Task 1: Merge Flight Performance Data ---
        perf_pattern = 'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_*.csv'
        perf_output = 'report_carrier_merged.csv'
        perf_expected = [
            'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_3.csv',
            'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_6.csv',
            'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_9.csv',
            'On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2022_12.csv'
        ]
        success1 = merge_files(data_dir, perf_pattern, perf_output, perf_expected)
        if not success1:
            overall_success = False # Mark overall failure if this step fails


        # --- Task 2: Merge Weather Data ---
        weather_pattern = 'top10airport_weather_*.csv' # More general pattern to catch all relevant files
        weather_output = 'top10_weather_merged.csv'
        weather_expected = [
            'top10airport_weather_202203.csv',
            'top10airport_weather_202206.csv',
            'top10airport_weather_202209.csv',
            'top10airport_weather_202212.csv'
        ]
        # Ensure we don't accidentally merge the output file if the script is run twice
        # A more robust way is to filter the glob result, but checking the pattern is usually sufficient
        if weather_output in weather_pattern:
             print(f"Warning: Output filename '{weather_output}' might match input pattern '{weather_pattern}'. Adjust pattern if needed.")

        success2 = merge_files(data_dir, weather_pattern, weather_output, weather_expected)
        if not success2:
            overall_success = False # Mark overall failure if this step fails

        # --- Final Summary ---
        print("\n--- Merge Summary ---")
        if success1:
            print(f"Flight Performance Data merge: SUCCESS -> {perf_output}")
        else:
            print(f"Flight Performance Data merge: FAILED")

        if success2:
            print(f"Weather Data merge: SUCCESS -> {weather_output}")
        else:
            print(f"Weather Data merge: FAILED")

        if not overall_success:
             print("\nOne or more merge tasks failed.")
             sys.exit(1) # Exit with error code if any merge failed
        else:
             print("\nAll merge tasks completed successfully.")


    except Exception as e:
        print(f"\nAn unexpected error occurred during the merge process: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- Run the function when the script is executed ---
if __name__ == "__main__":
    run_all_merges()
