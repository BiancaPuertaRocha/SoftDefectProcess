
# Orientation

## 1st step
Collect the data from GitHub (API), SonarScanner and SDPTools. Use the orientation present in data_collection/readme.md.

## 2nd step 
Test all the Feature Selection methods. Use the preprocess/readme.md orientation. This will save the datasets with the selected features in preprocess/data/datasets and the logs from the processing in preprocess/data/logs with some metrics extracted.
 
Also in feature_selection/readme.md you will find guidance to build and test models without any preprocessing. The results will be saved at feature_selection/data/logs

## 3th step 
Thest all the feature selection and data balance methods with each other using preprocess/readme.md orientation. The resulted datasets will be saved at preprocess/data/datasets with the datasets from each test arrangement and the evaluation will be saved at preprocess/data/logs.

At this point, we will have the best combination of feature selection and data balance to each machine learning algorithm (RF, Bagging with RF, Bagging with DT, Voting) and to each project (Dubbo, Iceberg, Pinot) and all the logs with metrics.

