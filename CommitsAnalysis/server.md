# Config server

## Run sonarQube
```
docker-compose up
```

## Create project sonar

```
curl -u "admin:admin" -X POST "http://localhost:9001/api/projects/create" \
  -d "name=flink" \
  -d "project=flink"
```

## Install sonnar-scanner (docker)
```
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip
unzip sonar-scanner-cli-4.8.0.2856-linux.zip
sudo mv sonar-scanner-4.8.0.2856-linux /opt/sonar-scanner
sudo ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner

```

## Create dir to store the repos

```
  mkdir repos
```

# Run

## Step 1:

```
  python commits_extract_1.py --repo_owner apache --repo_name airflow --output_csv commits_data.csv
```

## Step 2:
```
  python sonar_extract_2.py --output_csv sonarqube_metrics.csv --sonar_project_key dubbo --branch 3.3 --sonar_sources src
```