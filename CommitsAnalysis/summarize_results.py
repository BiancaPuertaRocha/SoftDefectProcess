import os
import pandas as pd

main_folder = 'data/logs'

metric_columns = [
    'auc', 'accuracy', 'precision', 'recall', 'f1',
    'precision_0', 'recall_0', 'f1_0',
    'precision_1', 'recall_1', 'f1_1',
    'train_time_ms', 'predict_time_ms', 'tunning_and_preprocess_time_ms'
]

# Loop through first-level folders
for first_level_folder in os.listdir(main_folder):
    first_level_path = os.path.join(main_folder, first_level_folder)

    if os.path.isdir(first_level_path):
        print(f'Entering folder: {first_level_folder}')
        
        summary_rows = []  # List to collect all summaries for this first-level folder
        
        # Loop through second-level subfolders
        for second_level_folder in os.listdir(first_level_path):
            second_level_path = os.path.join(first_level_path, second_level_folder)

            if os.path.isdir(second_level_path):
                print(f'  Processing subfolder: {second_level_folder}')
                
                # Collect CSVs in sub-subfolder
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
                    mean_metrics = combined_df[metric_columns].mean()
                    mean_metrics['subfolder'] = second_level_folder
                    summary_rows.append(mean_metrics)
                else:
                    print(f'    No CSV files found in {second_level_path}')

        # Save all summaries together (if any) to a CSV
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = os.path.join(first_level_path, f'{first_level_folder}_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f'  Saved combined summary to: {summary_csv_path}')
        else:
            print(f'  No data to summarize in {first_level_folder}')