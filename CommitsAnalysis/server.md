# Config server

First, make sure your local git version is 2.43.0 and your python version is 3.9.6 (we had problems with the git and python versions).

## Run sonarQube (inside this repo)
```
docker-compose up
```

## Create your sonar project

```
curl -u "admin:admin" -X POST "http://localhost:9001/api/projects/create" \
  -d "name=project_name" \
  -d "project=project_name"
```

## Install sonnar-scanner 
```
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip
unzip sonar-scanner-cli-4.8.0.2856-linux.zip
sudo mv sonar-scanner-4.8.0.2856-linux /opt/sonar-scanner
sudo ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner

```

## Create dir to store the repos and sonar_results

```
  mkdir repos
  mkdir sonar_results
```

## Create your virtual environment 
```
  python -m venv venv
  source ./venv/bin/activate
  pip install -r requirements 
```

# Run

## Step 0:
Use SDPTool to extract issue and pull request data. 

## Step 1:

```
  python commits_extract_1.py --repo_owner owner_name --repo_name project_name --output_csv commits_data.csv
```

## Step 2:
Run sonar scanner in each commit to extract modifications (files): code smells and bugs.
```
  python sonar_extract_2.py --output_csv sonarqube_metrics_project_name_final.csv --output_csv_bugs bugs_project_name_final.csv --output_csv_smells smells_project_name_final.csv --sonar_project_key project_name --branch main --sonar_sources . --start_sha fd4688cf603a713c578ab2f54d403daa922e2ee3 (optional)

```

## Step 3: 
Compile data about fix modifications.
```
  python compile_files_sdptool.py --folder issues_by_sdptool/project_name
```

## Step 4:
Compile files extracted by sonar scaner
```
python compile_files_sonar.py dubbo /caminho/para/pasta
```

# About the data 

## repos
The repositories that we will study.

## commits_from_api
Commits extracte with API requests.

## issues_by_sdp_tool
These are the databases extracted using SDPTool. They include data about the issues and pull request of issues so we can know which commits are the bug fixing (ans after we will be able to compare to find the bug introducing)

The compile_files_issue.py is the code that compiles this info.