import wandb
import pandas as pd
import os
import joblib
from config import WandbSettings


def download_data():
    # os.system(f"wandb login --relogin {WandbSettings().wandb_api}")
    run = wandb.init(project="project_heart")
    artifact = run.use_artifact("project_heart/heart_2020_cleaned.csv:latest")
    return pd.read_csv(artifact.file())


def download_model():
    # os.system(f"wandb login --relogin {WandbSettings.WANDB_API}")
    run = wandb.init(project="project_heart")
    artifact = run.use_artifact(
        "diego25rm/project_heart/model_export:v0", type="pipeline_artifact"
    )
    return joblib.load(artifact.file())
