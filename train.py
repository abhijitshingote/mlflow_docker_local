import configparser
import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import psycopg2 as psy

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# config = configparser.ConfigParser()
# config.read('config.ini')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


experiment_name='LinearRegression'
remote_server_uri = "http://127.0.0.1:5011"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment_name)

experiment = mlflow.get_experiment_by_name(experiment_name)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
if __name__ == '__main__':
    print('Hello Radhika!')
    # mlflow.set_tracking_uri("http://localhost:5000")

    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        print(f"tracking_uri={mlflow.get_tracking_uri()}")
        print(f"artifact_uri={mlflow.get_artifact_uri()}")
        mlflow.sklearn.log_model(lr, "model")

        # print(f"run_id={run_id}")
