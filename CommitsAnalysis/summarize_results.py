"""
When running the all_fs_db.py algorithm, this script is used to summarize the results.
"""
import os
import pandas as pd

main_folder = 'data/logs'

# Columns to average
metric_columns = [
    'auc', 'accuracy', 'precision', 'recall', 'f1',
    'precision_0', 'recall_0', 'f1_0',
    'precision_1', 'recall_1', 'f1_1',
    'train_time_ms', 'predict_time_ms', 'tunning_and_preprocess_time_ms'
]

# Columns to keep (including fs and balancer)
group_columns = ['fs', 'balancer']

# Loop through first-level folders
for first_level_folder in os.listdir(main_folder):
    first_level_path = os.path.join(main_folder, first_level_folder)

    if os.path.isdir(first_level_path):
        print(f'Entering folder: {first_level_folder}')
        
        summary_rows = []  # Store results for this folder
        
        # Loop through subfolders (second level)
        for second_level_folder in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level_folder)

            if os.path.isdir(second_level_path):
                print(f'  Processing subfolder: {second_level_folder}')
                
                # Collect CSV files
                csv_files = [
                    os.path.join(second_level_path, f)
                    for f in os.listdir(second_level_path)
                    if f.endswith('.csv')
                ]
                
                all_data = []
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        all_data.append(df)
                    except Exception as e:
                        print(f'    Error reading {csv_file}: {e}')
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)

                    # Group by fs and balancer, then compute mean of metrics
                    grouped = combined_df.groupby(group_columns)[metric_columns].mean().reset_index()

                    # Add subfolder column
                    grouped['subfolder'] = second_level_folder

                    # Append to overall summary
                    summary_rows.append(grouped)
                else:
                    print(f'    No CSV files found in {second_level_path}')

        # Save combined summary
        if summary_rows:
            final_summary_df = pd.concat(summary_rows, ignore_index=True)

            # Reorder columns: fs, balancer, subfolder, metrics...
            final_columns = group_columns + ['subfolder'] + metric_columns
            final_summary_df = final_summary_df[final_columns]

            summary_csv_path = os.path.join(first_level_path, f'{first_level_folder}_summary.csv')
            final_summary_df.to_csv(summary_csv_path, index=False)
            print(f'  Saved combined summary to: {summary_csv_path}')
        else:
            print(f'  No data to summarize in {first_level_folder}')