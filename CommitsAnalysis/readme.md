
# Orientation

## 1st step
Collect the data from GitHub (API), SonarScanner and SDPTools. Use the orientation present in data_collection/readme.md.

## 2nd step 
Test all the Feature Selection mathods. Use the feature_selection/readme.md orientation. This will save the datasets with the selected features in feature_selection/data/datasets and the logs from the processing in feature_selection/data/logs with some metrics extracted.

## 3rd step 
Test all the data balance methods. Use data_balance/readme.md orientation. The resulted datasets will be saved at data_balance/data/datasets with the balanced datasets to each method and the evaluation will be saved at data_balance/data/logs.

## 4th step 
Thest all the feature selection and data balance methods with each other using preprocess/readme.md orientation. The resulted datasets will be saved at preprocess/data/datasets with the datasets from each test arrangement and the evaluation will be saved at preprocess/data/logs.

At this point, we will have the best combination of feature selection and data balance to each machine learning algorithm (RF, Bagging with RF, Bagging with DT, Voting) and to each project (Dubbo, Iceberg, Pinot).

## 5th step
Construct a model and test with this data to each project with the best performing combination using... ESCREVER MAIS AQUI