from preprocesamiento import transformar_entrada
from model_loader import cargar_modelo_por_numero

def predecir(data: dict, modelo_numero: int):
    # Cargar el modelo correcto
    modelo = cargar_modelo_por_numero(modelo_numero)

    # Transformar entrada
    entrada_modelo = transformar_entrada(data)

    # Realizar predicción
    pred = modelo.predict(entrada_modelo)[0]

    return {
        "modelo_usado": (
            "Random Forest" if modelo_numero == 1 else
            "XGBoost" if modelo_numero == 2 else
            "LightGBM"
        ),
        "entrada": data,
        "prediccion_total_alquileres": int(pred)
    }
