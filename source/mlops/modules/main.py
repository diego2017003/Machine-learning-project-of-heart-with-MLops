from source.mlops.modules.download import *
from source.mlops.modules.preprocessing import *
from source.mlops.modules.train import *
import wandb


def main():
    data = download_data()
    data_resample = Preprocessing_initial_data(
        target_column="HeartDisease"
    ).fit_transform(data)
    pipeline = Model_pipeline()
    y = LabelEncoder().fit_transform(
        data_resample["HeartDisease"].values.reshape(-1, 1)
    )
    y = pd.DataFrame(y, columns=["HeartDisease"])

    X = data_resample.drop(columns=["HeartDisease"])
    pipeline.fit(X, y)
    model_artifact = "model_export"
    joblib.dump(pipeline, model_artifact)
    artifact = wandb.Artifact(
        model_artifact,
        type="pipeline_artifact",
        description="A full pipeline composed of a Preprocessing Stage and a Decision Tree model",
    )
    run = wandb.init(project="project_heart", save_code=True)
    artifact.add_file(model_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":
    main()
