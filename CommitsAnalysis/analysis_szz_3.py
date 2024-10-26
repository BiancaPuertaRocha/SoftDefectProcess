import pandas as pd
import argparse
import time

def process_files(fixes_path, modifications_path, commits_path, output_path):
    # Load the CSV files
    fixes_df = pd.read_csv(fixes_path)  # CSV for issue-fixing modifications
    modifications_df = pd.read_csv(modifications_path)  # CSV for general modifications
    commits_df = pd.read_csv(commits_path)  # CSV with commit data

    # Remove prefix before ":" in file names in fixes and modifications DataFrames
    fixes_df['FILE_NAME'] = fixes_df['FILE_NAME'].str.split(':').str[-1]
    modifications_df['file'] = modifications_df['file'].str.split(':').str[-1]

    # Step 1: Merge the fixes file with the commits file
    fixes_commit_df = pd.merge(fixes_df, commits_df, left_on='MERGE_COMMIT_SHA', right_on='sha', suffixes=('', '_commit'))

    # Step 2: Merge the general modifications file with the commits file
    modifications_commit_df = pd.merge(modifications_df, commits_df, left_on='commit_sha', right_on='sha', suffixes=('', '_commit'))

    # Convert dates to datetime format for comparison operations
    fixes_commit_df['commit_date'] = pd.to_datetime(fixes_commit_df['commit_date'])
    modifications_commit_df['commit_date'] = pd.to_datetime(modifications_commit_df['commit_date'])

    # Step 3: Identify modifications prior to the closest fix modification
    failure_prone_modifications = []
    start_time = time.time()
    for index, row in fixes_commit_df.iterrows():
        # Filter for modifications before the fix and with a different SHA
        previous_modifications = modifications_commit_df[
            (modifications_commit_df['file'] == row['FILE_NAME']) &  # Same file
            (modifications_commit_df['commit_date'] < row['CREATED_AT']) &  # Earlier date
            (modifications_commit_df['commit_sha'] != row['MERGE_COMMIT_SHA'])  # Different SHA
        ]
        
        # Find the closest modification before the fix
        if not previous_modifications.empty:
            last_modification = previous_modifications.sort_values(by='commit_date').iloc[-1]
            # Mark as failure-prone if there is a previous closest modification
            last_modification['failure_prone'] = 1
            failure_prone_modifications.append(last_modification)

        # Print progress every 5 seconds
        if time.time() - start_time > 5:
            print(f"{index + 1} issue modifications processed, {len(failure_prone_modifications)} failure-prone modifications found.")
            start_time = time.time()

    # Concatenate the marked failure-prone modifications into a final DataFrame
    failure_prone_df = pd.DataFrame(failure_prone_modifications)

    # Save the resulting DataFrame to a new CSV
    failure_prone_df.to_csv(output_path, index=False)
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
