import pandas as pd
import logging
import os
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

def evaluar_interna(df: pd.DataFrame, save_path: str) -> pd.DataFrame:
    """
    Calcula métricas de evaluación interna para clustering.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas y columna 'cluster'.
        save_path (str): Carpeta donde guardar el HTML de resultados.

    Returns:
        pd.DataFrame: DataFrame con métricas internas calculadas.
    """
    X = df.drop(columns=['cluster'])
    labels = df['cluster']

    resultados = {
        'Silhouette Score': silhouette_score(X, labels),
        'Davies-Bouldin Index': davies_bouldin_score(X, labels),
        'Calinski-Harabasz Index': calinski_harabasz_score(X, labels)
    }

    df_result = pd.DataFrame(resultados.items(), columns=['Metrica', 'Valor'])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    ruta_html = os.path.join(save_path, "evaluacion_interna.html")
    df_result.to_html(ruta_html, index=False, float_format="{:.4f}".format)

    logging.info("Evaluación interna completada y exportada")
    return df_result

def restaurar_variable_clase(df_dummificado: pd.DataFrame, df_original: pd.DataFrame, prefijo: str = "clase_tipo_") -> pd.DataFrame:
    """
    Restaura la variable categórica original a partir de columnas dummificadas.

    Detecta automáticamente la clase de referencia (omitida en la dummificación)
    y reconstruye la columna original (por ejemplo, 'clase_tipo').

    Args:
        df_dummificado (pd.DataFrame): DataFrame con variables dummificadas.
        df_original (pd.DataFrame): DataFrame original que contiene la variable categórica antes de dummificar.
        prefijo (str): Prefijo común de las columnas dummificadas (default: 'clase_tipo_').

    Returns:
        pd.DataFrame: Mismo DataFrame con nueva columna restaurada (sin eliminar dummies).
    """
    # Detectar nombre real de la columna original
    nombre_col = prefijo.rstrip('_')

    # Obtener clases
    clases_originales = set(df_original[nombre_col].unique())
    clases_dummificadas = {col.replace(prefijo, "") for col in df_dummificado.columns if col.startswith(prefijo)}
    clases_faltantes = clases_originales - clases_dummificadas

    if len(clases_faltantes) != 1:
        raise ValueError(f"No se pudo identificar una única clase faltante. Detectadas: {clases_faltantes}")

    clase_faltante = list(clases_faltantes)[0]
    logging.info(f"Clase faltante detectada: '{clase_faltante}'")

    # Reconstrucción
    clase_cols = [col for col in df_dummificado.columns if col.startswith(prefijo)]
    df_resultado = df_dummificado.copy()

    df_resultado[nombre_col] = df_resultado[clase_cols].idxmax(axis=1).str.replace(prefijo, "", regex=False)
    mask_base = df_resultado[clase_cols].sum(axis=1) == 0
    df_resultado.loc[mask_base, nombre_col] = clase_faltante

    logging.info(f"Variable '{nombre_col}' restaurada correctamente en el DataFrame")
    return df_resultado

def evaluar_externa(df: pd.DataFrame, etiqueta_referencia: str, save_path: str) -> pd.DataFrame:
    """
    Calcula métricas de evaluación externa comparando clusters con etiquetas de referencia.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'cluster' y una columna de clase real.
        etiqueta_referencia (str): Nombre de la columna de etiquetas verdaderas.
        save_path (str): Carpeta donde se guardará el archivo HTML con los resultados.

    Returns:
        pd.DataFrame: DataFrame con métricas de evaluación externa.
    """
    # Verificaciones
    if 'cluster' not in df.columns:
        logging.error("La columna 'cluster' no existe en el DataFrame.")
        raise ValueError("La columna 'cluster' no existe en el DataFrame.")

    if etiqueta_referencia not in df.columns:
        logging.error(f"La columna de referencia '{etiqueta_referencia}' no se encuentra en el DataFrame.")
        raise ValueError(f"La columna de referencia '{etiqueta_referencia}' no existe en el DataFrame.")

    # Cálculo de métricas
    labels_true = df[etiqueta_referencia]
    labels_pred = df['cluster']

    resultados = {
        'Adjusted Rand Index': adjusted_rand_score(labels_true, labels_pred),
        'Normalized Mutual Information': normalized_mutual_info_score(labels_true, labels_pred),
        'Homogeneity Score': homogeneity_score(labels_true, labels_pred),
        'Completeness Score': completeness_score(labels_true, labels_pred),
        'V-measure': v_measure_score(labels_true, labels_pred)
    }

    df_result = pd.DataFrame(resultados.items(), columns=['Metrica', 'Valor'])

    # Guardar resultados
    Path(save_path).mkdir(parents=True, exist_ok=True)
    ruta_html = os.path.join(save_path, "evaluacion_externa.html")
    df_result.to_html(ruta_html, index=False, float_format="{:.4f}".format)

    logging.info(f"Evaluación externa completada y exportada a: {ruta_html}")
    return df_result
