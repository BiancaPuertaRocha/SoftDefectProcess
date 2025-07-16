
# Run all Feature Selection methods alone

## Run all the FS methods wit optimization and save the results (metrics) in data/logs/final_project_new_features_logs.csv
```
python preprocess/workers/just_fs.py data_collection/szz/final_project_new_features.csv run_all
```

## Run evaluation to just one method and with one model

Options:
 - run_fisher_random_forest
 - run_ga_random_forest
 - run_chi_random_forest
 - run_fisher_bagging_random_forest
 - run_ga_bagging_random_forest
 - run_chi_bagging_random_forest
 - run_fisher_bagging_cart
 - run_ga_bagging_cart
 - run_chi_bagging_cart
 - run_fisher_voting
 - run_ga_voting
 - run_chi_voting
```
python preprocess/workers/just_fs.py data_collection/szz/final_project_new_features.csv [option]

```

## Count the instances in each class
```
python preprocess/workers/just_fs.py data_collection/szz/final_project_new_features.csv run_no_preprocess --run_count
```

# Evaluate the bext combination of preprocessing (FS + data balance)

## Run all the combinations with one model construvtion algorithm and save the logs in data/logs/project__modelName__results.csv

Options: 
 - rf 
 - voting
 - bag_rf
 - bag_dt

```
python -m preprocess.workers.all_fs_db --input data_collection/szz/final_[project]_new_features.csv --model [option]
``` 
All the metrics are saved in data/logs/
The final datasets are saved in data/datasets/

<!-- /home/bianca/SoftDefectProcess/CommitsAnalysis
source venv/bin/activate
python -m preprocess.workers.all_fs_db --input data_collection/szz/final_dubbo_new_features.csv --model rf -->