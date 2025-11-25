import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib
import os

# dataset

df = pd.read_csv("datos/dataset_alquiler.csv")
df = df.dropna()  # eliminando filas nulas

# selección de caracteristicas con excepcion de las columnas u_casuales y u_registrados

caracteristicas = [
    "temporada", "anio", "mes", "hora", "feriado",
    "dia_semana", "dia_trabajo", "clima",
    "temperatura", "sensacion_termica",
    "humedad", "velocidad_viento"
]
X = df[caracteristicas]
y = df["total_alquileres"]

# separamos datos de entrenamiento y prueba

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# modelo RANDOM FOREST

print("entrenando modelo random forest")
modelo_rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

modelo_rf.fit(X_entrenamiento, y_entrenamiento)
pred_rf = modelo_rf.predict(X_prueba)

# metricas de desempeño
print("random forest error cuadratico medio MSE:", mean_squared_error(y_prueba, pred_rf))
print("random forest coeficiente de determinacion R2:", r2_score(y_prueba, pred_rf))

importancias_rf = pd.DataFrame({
    "caracteristica": caracteristicas,
    "importancia": modelo_rf.feature_importances_
}).sort_values(by="importancia", ascending=False)
print(importancias_rf)

joblib.dump(modelo_rf, "modelos/random_forest.pkl")
print("modelo random forest guardado")

# modelo XGBOOST

print("entrenando modelo xgboost")
modelo_xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(X_entrenamiento, y_entrenamiento)
pred_xgb = modelo_xgb.predict(X_prueba)

print("xgboost error cuadratico medio MSE:", mean_squared_error(y_prueba, pred_xgb))
print("xgboost coeficiente de determinacion R2:", r2_score(y_prueba, pred_xgb))

importancias_xgb = pd.DataFrame({
    "caracteristica": caracteristicas,
    "importancia": modelo_xgb.feature_importances_
}).sort_values(by="importancia", ascending=False)
print(importancias_xgb)

joblib.dump(modelo_xgb, "modelos/xgboost.pkl")
print("modelo xgboost guardado")

# modelo LIGHTGBM

print("entrenando modelo lightgbm")
modelo_lgb = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

modelo_lgb.fit(X_entrenamiento, y_entrenamiento)
pred_lgb = modelo_lgb.predict(X_prueba)

print("lightgbm error cuadratico medio MSE:", mean_squared_error(y_prueba, pred_lgb))
print("lightgbm coeficiente de determinacion R2:", r2_score(y_prueba, pred_lgb))

importancias_lgb = pd.DataFrame({
    "caracteristica": caracteristicas,
    "importancia": modelo_lgb.feature_importances_
}).sort_values(by="importancia", ascending=False)
print(importancias_lgb)

joblib.dump(modelo_lgb, "modelos/lightgbm.pkl")
print("modelo lightgbm guardado")

print("modelos entrenados correctamente")
