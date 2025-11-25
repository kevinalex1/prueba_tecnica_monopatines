from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predictor import predecir

app = FastAPI(
    title="API de prediccion de alquileres de monopatines",
    description="API para predecir el total de monopatines alquilados por hora usando los modelos XGBoost y LightGBM",
    version="1.0.0"
)

MODELOS = {
    1: "XGBOOST",
    2: "LIGHTGBM"
}

class PrediccionEntrada(BaseModel):
    temporada: str
    anio: int
    mes: int
    hora: int
    feriado: int
    dia_semana: int
    dia_trabajo: int
    clima: str
    temperatura: float
    sensacion_termica: float
    humedad: float
    velocidad_viento: float
    modelo: Optional[int] = 1 

@app.get("/health")
def health():
    return {"status": "ok", "message": "API funcionando"}

@app.post("/predict")
def predict(data: PrediccionEntrada):
    if data.modelo not in MODELOS:
        raise HTTPException(status_code=400, detail="modelo no disponible. Opciones: 1=XGBoost, 2=LightGBM")

    entrada_dict = data.dict()
    modelo_num = entrada_dict.pop("modelo")

    try:
        resultado = predecir(entrada_dict, modelo_num)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

    return {
        "modelo_usado": resultado["modelo_usado"],
        "entrada": resultado["entrada"],
        "prediccion_total_alquileres": resultado["prediccion_total_alquileres"]
    }
