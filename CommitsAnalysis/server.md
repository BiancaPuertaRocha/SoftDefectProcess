# Config server

## Run sonarQube
```
docker-compose up
```

## Generate Token
```
curl -X POST -u "admin:admin" \
"http://127.0.0.1:9001/api/user_tokens/generate" \
-d "name=sonnar_token"

```
substituir token em sonar_scanner_commits_2.py

## Create project sonar qube

```
curl -u "admin:admin" -X POST "http://localhost:9001/api/projects/create" \
  -d "name=dubbo" \
  -d "project=dubbo"
```

## Install sonnar-scanner (docker)
```
wget https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.8.0.2856-linux.zip
unzip sonar-scanner-cli-4.8.0.2856-linux.zip
sudo mv sonar-scanner-4.8.0.2856-linux /opt/sonar-scanner
sudo ln -s /opt/sonar-scanner/bin/sonar-scanner /usr/local/bin/sonar-scanner

```

## Run SonnarScanner (docker)
```
docker run \
  --rm \
  -e SONAR_HOST_URL="https://your-sonarqube-server" \
  -e SONAR_LOGIN="your-token" \
  -v $(pwd):/usr/src \
  sonarsource/sonar-scanner-cli


```