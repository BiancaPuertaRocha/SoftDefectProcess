# Evaluate the bext combination of preprocessing (FS + data balance)

## Run all the combinations with one model construvtion algorithm 
model options: 'rf', 'voting', 'bag_rf', 'bag_dt'
```
python -m preprocess.main --input data_collection/szz/final_dubbo_new_features.csv --model rf
``` 
All the metrics are saved in data/logs/
The final datasets are saved in data/datasets/