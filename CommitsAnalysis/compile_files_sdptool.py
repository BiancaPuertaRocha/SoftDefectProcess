import pandas as pd
import os
import argparse

def compile_data(folder):
    # Get files
    issues_file = os.path.join(folder, 'issue.csv')
    pull_requests_file = os.path.join(folder, 'pull_rq.csv')
    pr_files_file = os.path.join(folder, 'pr_files_changed.csv')

    issues_df = pd.read_csv(issues_file)
    pull_requests_df = pd.read_csv(pull_requests_file)
    pr_files_df = pd.read_csv(pr_files_file)

    # Merges based on ISSUE_ID
    merged_df = pd.merge(pr_files_df, pull_requests_df, on='ISSUE_ID', how='inner')  # Inner join to maintain just the ones tha have the reationship
    final_df = pd.merge(merged_df, issues_df, on='ISSUE_ID', how='inner')  # Inner join 

    final_df = final_df.drop_duplicates()

    # Save resultss
    if not final_df.empty:
        output_file = os.path.join(folder, 'compiled_results.csv')
        final_df.to_csv(output_file, index=False)
        print(f"All info compiled '{output_file}'.")
    else:
        print("No data found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compile issue, pull request, and file change data from CSVs.')
    parser.add_argument('--folder', required=True, help='Folder containing the CSV files.')

    args = parser.parse_args()
    compile_data(args.folder)
