# Evaluate the bext combination of preprocessing (FS + data balance)

## Run all the combinations with one model construvtion algorithm 
model options('rf', 'voting', 'bag_rf', 'bag_dt')
```
python main.py --input file_path --model model_name
``` 
All the metrics are saved in data/logs/results.csv.
The final datasets are saved in data/datases/file_name.csv