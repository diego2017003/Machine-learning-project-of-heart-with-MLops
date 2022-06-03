import wandb
import pandas as pd
import os
import joblib
from config import WandbSettings


def download_data():
    """download the data from wandb

    Returns:
        _type_: dataset or file coming from outer programa
    """
    # os.system(f"wandb login --relogin ")# relogin must be made with secretes and environment variable on config file
    run = wandb.init(project="project_heart")
    artifact = run.use_artifact("project_heart/heart_2020_cleaned.csv:latest")
    return pd.read_csv(artifact.file())


def download_model():
    """download of the model from wandb

    Returns:
        _type_: dataset or file coming from outer program
    """
    # os.system(f"wandb login --relogin ")
    run = wandb.init(project="project_heart")
    artifact = run.use_artifact(
        "diego25rm/project_heart/model_export:v0", type="pipeline_artifact"
    )
    return joblib.load(artifact.file())
