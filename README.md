# Prediccion de alquiler de monopatines en Berlin, Alemania

## Descripcion

Esta API predice la cantidad de monopatines alquilados por hora en Berlín usando modelos lightgbm y xgboost

Incluye el pipeline modular para preprocesamiento y prediccion, y una API FastAPI para realizar predicciones.

## Estructura del proyecto

prueba_monopatines/
├── datos/ # Dataset
│ └── dataset_alquiler.csv
├── modelos/ # Modelos entrenados (.pkl)
│ ├── xgboost.pkl
│ └── lightgbm.pkl
├── api.py # API FastAPI
├── predictor.py # Función de predicción
├── model_loader.py # Carga modelos
├── preprocesamiento.py# Transformaciones de datos
├── requirements.txt # Dependencias
└── README.md # Este archivo

## Instalacion y datos de entrenamiento

1. Clonar el repositorio:

git clone https://github.com/kevinalex1/prueba_tecnica_monopatines.git

Instalar dependencias python 3.11:

pip install -r requirements.txt
Ejecutar la API localmente:

uvicorn api:app --reload
La API para probar las predicciones estara en:

http://127.0.0.1:8000/docs

Desempeño de los modelos

Los modelos se entrenaron con 70% de los datos para entrenamiento y 30% para pruebas:

Modelo R² en test
XGBoost 0.94
LightGBM 0.95

Endpoints que verifican

GET /predict_random

Genera una predicción usando una fila aleatoria del dataset, mostrando solo las columnas de entrada (sin u_casuales ni u_registrados) y con los valores categoricos en formato de entrada para el modelo elegido

Parametro a ingresar (1 o 2):

modelo: 1 - XGBoost, 2 - LightGBM (por defecto estara 2)

POST /predict
Recibe datos de entrada y devuelve la predicción de alquileres.

JSON de ejemplo:

{
"temporada": "verano",
"anio": 2013,
"mes": 7,
"hora": 17,
"feriado": 0,
"dia_semana": 4,
"dia_trabajo": 1,
"clima": "soleado",
"temperatura": 0.9,
"sensacion_termica": 0.92,
"humedad": 0.4,
"velocidad_viento": 0.1,
"modelo": 2
}

Modelos disponibles:

1 XGBoost

2 LightGBM

Ejemplo de respuesta:

{
"modelo_usado": "LightGBM",
"entrada": { ...datos de entrada... },
"prediccion_total_alquileres": 134
}

Despliegue
La API está disponible online en Render:
https://prueba-tecnica-monopatines.onrender.com/docs

Se puede probar desde Swagger UI (/docs)

Notas
Se elimino el modelo pesado (Random Forest) para mantener el repositorio ligero.
Solo se incluyen XGBoost y LightGBM.
Los archivos .pkl están en la carpeta modelos/.

Dependencias
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
