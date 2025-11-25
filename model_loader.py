import joblib

def cargar_modelo_por_numero(n: int):
    if n == 1:
        ruta = "modelos/random_forest.pkl"
    elif n == 2:
        ruta = "modelos/xgboost.pkl"
    elif n == 3:
        ruta = "modelos/lightgbm.pkl"
    else:
        raise ValueError("Elija un modelo válido: 1, 2 o 3")

    modelo = joblib.load(ruta)
    return modelo
