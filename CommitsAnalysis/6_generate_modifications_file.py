import pandas as pd
import argparse

def unwind_and_merge(input_file1, input_file2, output_file):
    # Load the first CSV file and perform the unwind on modified_files
    df1 = pd.read_csv(input_file1)
    # df1_expanded = df1.assign(file=df1['modified_files'].str.split(',')).explode('file')
    df1.drop(columns=['modified_files'], inplace=True)

    # Load the second CSV file
    df2 = pd.read_csv(input_file2)

    df1['FILE_NAME'] = df1['file'].str.split(':').str[-1]
    df2['file'] = df2['file'].str.split(':').str[-1]

    # Merge on sha and file
    merged_df = pd.merge(df1, df2, left_on=['sha', 'file'], right_on=['commit_sha', 'file'], how='left')

    # Save the merged DataFrame to the output file
    merged_df.to_csv(output_file, index=False)
    print(f"Merge completed. Output saved to {output_file}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Unwind modified files and merge with another CSV by sha and file.")
    parser.add_argument("--commits", required=True, help="Path to the first input CSV file (with modified_files column)")
    parser.add_argument("--sonar_metrics", required=True, help="Path to the second input CSV file (with bug and code smell data)")
    parser.add_argument("--output", required=True, help="Path to the output CSV file")

    args = parser.parse_args()

    # Run the unwind and merge function
    unwind_and_merge(args.commits, args.sonar_metrics, args.output)
