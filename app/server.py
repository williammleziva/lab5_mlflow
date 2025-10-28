# app/server.py
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    class_id = []
    class_label = []

    for sample in req.samples:
        model_input = [[
            sample.sepal_length,
            sample.sepal_width,
            sample.petal_length,
            sample.petal_width
        ]]
        prediction = model.predict(model_input)
        print(f"Predicted class w/ version {MODEL_VERSION} for sample {sample}: {prediction[0]}")

        class_id.append(int(prediction[0]))
        class_label.append(IRIS_LABELS[int(prediction[0])])

    return PredictResponse(
        class_id=class_id,
        class_label=class_label
    )
    
@app.get("/version", tags=["version"], summary="Get model serving version")
def get_version():
    return {"version": MODEL_VERSION} 


class SetVersionRequest(BaseModel):
    version: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"version": "2"}
            ]
        }
    }

class VersionResponse(BaseModel):
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Model version updated to 2"}
            ]
        }
    }

@app.post("/version", tags=["version"], summary="Set model serving version",
          response_model=VersionResponse)
def set_version(req: SetVersionRequest):
    global MODEL_VERSION, MODEL_URI, model
    MODEL_VERSION = req.version
    MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return {"message": f"Model version updated to {MODEL_VERSION}"}


# TODO Add endpoint to get the current model serving version
# TODO Add endpoint to update the serving version
# TODO Predict using the correct served version
