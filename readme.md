1. git clone https://github.com/abhijitshingote/mlflow_docker_local.git 
2. cd mlflow_docker_local
3. build image and launch services -  docker-compose build --no-cache && docker-compose up
4. run experiment from local conda environment - python train.py
5. check results at - http://127.0.0.1:5011
