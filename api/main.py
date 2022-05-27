from fastapi import FastAPI
from pydantic import BaseModel
import pickle

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


with open("./finalized_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


@app.post("/prediction")
def get_potability(data: heart_metrics):
    received = data.dict()
    ph = received["ph"]
    Hardness = received["Hardness"]
    Solids = received["Solids"]
    Chloramines = received["Chloramines"]
    Sulfate = received["Sulfate"]
    Conductivity = received["Conductivity"]
    Organic_carbon = received["Organic_carbon"]
    Trihalomethanes = received["Trihalomethanes"]
    Turbidity = received["Turbidity"]
    pred_name = loaded_model.predict(
        [
            [
                ph,
                Hardness,
                Solids,
                Chloramines,
                Sulfate,
                Conductivity,
                Organic_carbon,
                Trihalomethanes,
                Turbidity,
            ]
        ]
    ).tolist()[0]
    return {"Prediction": pred_name}
