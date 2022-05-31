from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import wandb
import pandas as pd
import source.mlops.modules.train
import source.mlops.modules.preprocessing

app = FastAPI()

# Creating a class for the attributes input to the ML model.
class Heart_metrics(BaseModel):
    BMI: float
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Race: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str
    SleepTime: float
    Asthma: str
    KidneyDisease: str
    SkinCancer: str

    class config:
        schema_extra = {
            "example": {
                "BMI": 16.4,
                "Smoking": "Yes",
                "AlcoholDrinking": "No",
                "Stroke": "No",
                "PhysicalHealth": 3.0,
                "MentalHealth": 30.0,
                "DiffWalking": "No",
                "Sex": "Female",
                "AgeCategory": "55-59",
                "Race": "White",
                "Diabetic": "Yes",
                "PhysicalActivity": "Yes",
                "GenHealth": "Very good",
                "SleepTime": 5.0,
                "Asthma": "Yes",
                "KidneyDisease": "No",
                "SkinCancer": "Yes",
            }
        }


run = wandb.init(project="project_heart", job_type="api")


@app.get("/")
def home():
    return {"Hello": "World"}


@app.post("/prediction")
def heart_prediciton(data: Heart_metrics):
    artifact = run.use_artifact(
        "diego25rm/project_heart/model_export:v1", type="pipeline_artifact"
    ).file()
    loaded_model = joblib.load(artifact)
    recieve = dict(data)
    X = pd.DataFrame([data])
    print(X)
    prediction = loaded_model.predict(X)
    return {"Prediction": data}
