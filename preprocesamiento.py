import pandas as pd

def transformar_entrada(data: dict):
    df = pd.DataFrame([data])

    # Mapeos categóricos necesarios
    mapa_temporada = {
        "invierno": 0, "primavera": 1, "verano": 2, "otoño": 3
    }
    mapa_clima = {
        "soleado": 0, "nublado": 1, "lluvioso": 2, "nevado": 3
    }

    df["temporada"] = df["temporada"].map(mapa_temporada)
    df["clima"] = df["clima"].map(mapa_clima)

    # Mantener orden correcto de columnas
    columnas = [
        "temporada", "anio", "mes", "hora", "feriado",
        "dia_semana", "dia_trabajo", "clima",
        "temperatura", "sensacion_termica",
        "humedad", "velocidad_viento"
    ]

    return df[columnas]
