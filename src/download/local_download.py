from functools import partial
import os
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from config import config

def main(run_id):
    """Function to create and download artifacts from MLflow.
    
    The MLflow script for experiments is where this module is executed without issues.
    example: root_dir/mlflow_experiment.py, run the module in mlflow_experiment.py"""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient(config.TRACKING_URI)
    experiment_id = client.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id
    root_dir = os.path.realpath(f'../artifacts')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    list = ("0/default", f"{experiment_id}/{run_id}")
    concat_path = partial(os.path.join, root_dir)
    make_dir = partial(os.makedirs, exist_ok=True)
    for path_items in map(concat_path, list):
        make_dir(path_items)
    client.download_artifacts(run_id=run_id, path="", dst_path=os.path.realpath(f'../artifacts/{experiment_id}/{run_id}'))
    print(f"Model with run id: {run_id} has been successfully downloaded.")

if __name__ == "__main__":
    main()