import pandas as pd
import argparse
import time

def process_files(fixes_path, modifications_path, commits_path, output_path):
    failure_prone = 0

    # Load the CSV files
    fixes_df = pd.read_csv(fixes_path)  # CSV for issue-fixing modifications
    modifications_df = pd.read_csv(modifications_path)  # CSV for general modifications
    commits_df = pd.read_csv(commits_path)  # CSV with commit data

    print(f'initial modifications {len(modifications_df)}')

    # Remove prefix before ":" in file names in fixes and modifications DataFrames
    fixes_df['FILE_NAME'] = fixes_df['FILE_NAME'].str.split(':').str[-1]
    modifications_df['file'] = modifications_df['file'].str.split(':').str[-1]

    # Step 1: Merge the fixes file with the commits file
    fixes_commit_df = pd.merge(fixes_df, commits_df, left_on='MERGE_COMMIT_SHA', right_on='sha', suffixes=('', '_commit'), how='left')

    # Step 2: Merge the general modifications file with the commits file
    modifications_commit_df = pd.merge(modifications_df, commits_df, left_on='commit_sha', right_on='sha', suffixes=('', '_commit'), how='left')

    print(f'after merge modifications {len(modifications_commit_df)}')

    # Convert dates to datetime format for comparison operations
    fixes_commit_df['commit_date'] = pd.to_datetime(fixes_commit_df['commit_date'])
    modifications_commit_df['commit_date'] = pd.to_datetime(modifications_commit_df['commit_date'])
    fixes_commit_df['CREATED_AT'] = pd.to_datetime(fixes_commit_df['CREATED_AT'] / 1000, unit='s', utc=True)

    # Add a new column for failure-prone modifications and initialize with 0
    modifications_commit_df['failure_prone'] = 0

    # Step 3: Identify modifications prior to the closest fix modification
    start_time = time.time()
    for index, row in fixes_commit_df.iterrows():
        # Filter for modifications before the fix and with a different SHA
        previous_modifications = modifications_commit_df[
            (modifications_commit_df['file'] == row['FILE_NAME']) &  # Same file
            (modifications_commit_df['commit_date'] < row['CREATED_AT']) &  # Earlier date
            (modifications_commit_df['sha'] != row['MERGE_COMMIT_SHA'])  # Different SHA
        ]
        
        # Find the closest modification before the fix
        if not previous_modifications.empty:
            last_modification_index = previous_modifications['commit_date'].idxmax()
            # Mark the closest modification as failure-prone
            modifications_commit_df.at[last_modification_index, 'failure_prone'] = 1
            failure_prone += 1


        # Print progress every 5 seconds
        if time.time() - start_time > 5:
            print(f"{index + 1} issue modifications processed. {failure_prone} failure prone modifications found.")
            start_time = time.time()

    # Save the DataFrame with all modifications and failure-prone labels to a new CSV
    modifications_commit_df.to_csv(output_path, index=False)
    print("Processing completed. Results saved to:", output_path)

if __name__ == "__main__":
    # Configure argparse to accept input and output files
    parser = argparse.ArgumentParser(description="Process CSVs to identify failure-prone modifications.")
    parser.add_argument("--fixes", required=True, help="Path to the issue fixes CSV file")
    parser.add_argument("--modifications", required=True, help="Path to the general modifications CSV file")
    parser.add_argument("--commits", required=True, help="Path to the commits data CSV file")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")

    args = parser.parse_args()
    
    # Call the function with the arguments
    process_files(args.fixes, args.modifications, args.commits, args.output)
