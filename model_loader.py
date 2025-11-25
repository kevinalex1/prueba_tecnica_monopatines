import joblib
import os

def cargar_modelo_por_numero(n: int):
    if n == 1:
        ruta = "modelos/xgboost.pkl"
    elif n == 2:
        ruta = "modelos/lightgbm.pkl"
    else:
        raise ValueError("modelo. opciones: 1=XGBoost, 2=LightGBM")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"no se encontro el modelo")

    modelo = joblib.load(ruta)
    return modelo
