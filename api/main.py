from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import wandb
import pandas as pd

app = FastAPI()

# Creating a class for the attributes input to the ML model.
class heart_metrics(BaseModel):
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
    # Loading the trained model


run = wandb.init()
artifact = run.use_artifact(
    "diego25rm/project_heart/model_export:v1", type="pipeline_artifact"
)

loaded_model = joblib.load(artifact.file())


@app.post("/prediction")
def get_potability(data: heart_metrics):
    received = data.dict()
    X = pd.DataFrame(received)
    prediction = loaded_model.predict(X).tolist()[0]
    return {"Prediction": prediction}
