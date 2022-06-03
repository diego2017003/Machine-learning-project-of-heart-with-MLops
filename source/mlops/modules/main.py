from download import *
from preprocessing import *
from train import *
import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
)

logger = logging.getLogger()


def main():
    """this module builds the full pipeline model and exports it to the wandb project as a pipeline object"""
    logger.info("Downloading the data from wandb")
    data = download_data()

    logger.info(
        "apply a dwonsample in the data for equalize the target column proportion"
    )
    data_resample = Preprocessing_initial_data(
        target_column="HeartDisease"
    ).fit_transform(data)

    logger.info("create the full pipeline object")
    pipeline = Model_pipeline()

    logger.info("encoding the target variable and transform to DataFrame")
    y = LabelEncoder().fit_transform(
        data_resample["HeartDisease"].values.reshape(-1, 1)
    )
    y = pd.DataFrame(y, columns=["HeartDisease"])

    logger.info("drop the target from the raw data")
    X = data_resample.drop(columns=["HeartDisease"])

    logger.info("train the pipeline")
    pipeline.fit(X, y)

    logger.info("export the full pipeline model to the wandb project")
    model_artifact = "model_export"

    logger.info("encoding the pipeline model to a file in memory")
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
