# Select the Best Features

## RF to evaluate and tunning

### GA  
```
python main.py caminho/para/seu_arquivo.csv run_ga_random_forest
```
### Fisher Score
```
python main.py caminho/para/seu_arquivo.csv run_fisher_random_forest
```
### Chi Square 
```
python main.py caminho/para/seu_arquivo.csv run_chi_random_forest
```

## Bagging + RF to evaluate and tunning

### GA 
```
python main.py caminho/para/seu_arquivo.csv run_ga_bagging_random_forest
``` 

### Fisher Score
```
python main.py caminho/para/seu_arquivo.csv run_fisher_bagging_random_forest
``` 

### Chi Square 
```
python main.py caminho/para/seu_arquivo.csv run_chi_bagging_random_forest
``` 

## Bagging + DT to evaluate and tunning

### GA  
```
python main.py caminho/para/seu_arquivo.csv run_ga_bagging_cart
```
### Fisher Score
```
python main.py caminho/para/seu_arquivo.csv run_fisher_bagging_cart
```
### Chi Square 
```
python main.py caminho/para/seu_arquivo.csv run_chi_bagging_cart
```

## Voting to evaluate and tunning

### GA  
```
python main.py caminho/para/seu_arquivo.csv run_ga_voting
```
### Fisher Score
```
python main.py caminho/para/seu_arquivo.csv run_fisher_voting
```
### Chi Square 
```
python main.py caminho/para/seu_arquivo.csv run_chi_voting
```


## Execute all methods (RF, Bagging+RF, Bagging+DT, Voting)
```
python main.py caminho/para/seu_arquivo.csv run_all --skip_prepare (use if the dataset already passed by the message feature generation with TF-IDF - data_collection/9_new_features.py) 
```

## Build model without any preprocessing 
python main.py caminho/para/seu_arquivo.csv run_no_feature_selection