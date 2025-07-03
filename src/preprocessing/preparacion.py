import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Eliminar variables específicas
def eliminar_variables(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """
    Elimina variables específicas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        variables (list): Lista de nombres de columnas a eliminar.

    Returns:
        pd.DataFrame: DataFrame sin las variables indicadas.
    """
    df_filtrado = df.drop(columns=variables, errors='ignore')
    logging.info(f"Variables eliminadas: {variables}")
    return df_filtrado

# 2. Separar variables numéricas y categóricas
def separar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa un DataFrame en variables numéricas y categóricas.

    Args:
        df (pd.DataFrame): DataFrame completo a procesar.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con solo variables numéricas.
            - pd.DataFrame: DataFrame con solo variables categóricas.
    """
    df_num = df.select_dtypes(include=["number"]).copy()
    df_cat = df.select_dtypes(exclude=["number"]).copy()

    logging.info("Variables numéricas: %s", list(df_num.columns))
    logging.info("Variables categóricas: %s", list(df_cat.columns))

    return df_num, df_cat

# 3. Imputación basada en porcentaje de nulos
def generar_diagnostico_knn(df: pd.DataFrame, columna: str, k: int = 3, save_dir: str = "outputs/02_preparacion") -> pd.DataFrame:
    """
    Genera un diagnóstico detallado de la imputación por KNN para una columna con valores nulos.
    Para cada fila con nulo:
        - Identifica los K vecinos más cercanos (distancia euclidiana ignorando nulos)
        - Obtiene sus distancias, valores y calcula el valor imputado (media)

    Args:
        df (pd.DataFrame): DataFrame original (preferentemente numérico).
        columna (str): Nombre de la columna a imputar.
        k (int): Número de vecinos más cercanos a considerar.
        save_dir (str): Ruta donde se guardará el archivo HTML con el diagnóstico.

    Returns:
        pd.DataFrame: DataFrame resumen del diagnóstico, también guardado como HTML.
    """
    df_work = df.copy()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    distancias = nan_euclidean_distances(df_work)
    diagnostico = []

    for idx in df_work[df_work[columna].isnull()].index:
        dists = pd.Series(distancias[idx], index=df_work.index).drop(idx)
        vecinos = dists.nsmallest(k)
        valores_vecinos = df_work.loc[vecinos.index, columna]
        promedio = valores_vecinos.mean()

        fila_resultado = {
            "fila_con_nan": idx,
            "valor_imputado": round(promedio, 2)
        }
        for i, (vec_idx, dist) in enumerate(vecinos.items(), start=1):
            fila_resultado[f"vecino_{i}"] = vec_idx
            fila_resultado[f"dist_{i}"] = round(dist, 2)
            fila_resultado[f"val_{i}"] = df_work.at[vec_idx, columna]
        diagnostico.append(fila_resultado)

    df_diag = pd.DataFrame(diagnostico)
    ruta_html = os.path.join(save_dir, f"diagnostico_knn_{columna}.html")
    df_diag.to_html(ruta_html, index=False)

    return df_diag

def imputar_nulos(df: pd.DataFrame, k: int = 3, umbrales: dict = None, save_dir: str = "outputs/02_preparacion") -> pd.DataFrame:
    """
    Imputa valores nulos según el porcentaje de faltantes:
    - < bajo: elimina registros
    - entre bajo y medio: imputa con KNN
    - > medio: elimina variable
    También guarda un diagnóstico detallado por variable imputada.

    Args:
        df (pd.DataFrame): DataFrame original.
        k (int): Número de vecinos para KNN.
        umbrales (dict): Diccionario con claves "bajo" y "medio".
        save_dir (str): Carpeta donde guardar los diagnósticos.

    Returns:
        pd.DataFrame: DataFrame imputado.
    """
    if umbrales is None:
        umbrales = {"bajo": 0.03, "medio": 0.15}

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    total = len(df)
    df_resultado = df.copy()

    for col in df.columns:
        nulos = df[col].isnull().sum()
        porcentaje = nulos / total

        if nulos == 0:
            continue
        elif porcentaje < umbrales["bajo"]:
            df_resultado = df_resultado[df_resultado[col].notnull()]
            logging.info(f"{col}: Se eliminaron {nulos} filas con nulos (<3%)")
        elif porcentaje <= umbrales["medio"]:
            diagnostico = generar_diagnostico_knn(df_resultado.select_dtypes(include='number'), col, k)
            diagnostico_path = os.path.join(save_dir, f"diagnostico_knn_{col}.html")
            diagnostico.to_html(diagnostico_path, index=False)
            logging.info(f"{col}: Imputado con KNN. Diagnóstico guardado en {diagnostico_path}")

            imputer = KNNImputer(n_neighbors=k)
            df_resultado[[col]] = imputer.fit_transform(df_resultado[[col]])
        else:
            df_resultado = df_resultado.drop(columns=col)
            logging.info(f"{col}: Eliminada por tener >15% de nulos")

    return df_resultado

# 4. Detección de outliers univariados
def detectar_outliers_univariado(
    df_num: pd.DataFrame,
    ruta_salida: str = None
) -> pd.DataFrame:
    """
    Detecta outliers univariados en variables numéricas usando IQR.
    Opcionalmente guarda un resumen en HTML.

    Args:
        df_num (pd.DataFrame): DataFrame numérico
        ruta_salida (str): Ruta de salida para guardar HTML (opcional)

    Returns:
        pd.DataFrame: DataFrame booleano con True en posiciones de outliers
    """
    outliers = pd.DataFrame(False, index=df_num.index, columns=df_num.columns)
    resumen_outliers = []

    for col in df_num.columns:
        Q1 = df_num[col].quantile(0.25)
        Q3 = df_num[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_sup = Q3 + 1.5 * IQR
        limite_inf = Q1 - 1.5 * IQR

        outliers[col] = (df_num[col] < limite_inf) | (df_num[col] > limite_sup)
        cantidad = outliers[col].sum()

        logging.info(f"{col}: {cantidad} outliers univariados")
        resumen_outliers.append({"variable": col, "cantidad_outliers": cantidad})

    # Exportar resumen si se indica
    if ruta_salida:
        Path(ruta_salida).mkdir(parents=True, exist_ok=True)
        df_resumen = pd.DataFrame(resumen_outliers)
        df_resumen.to_html(os.path.join(ruta_salida, "outliers_univariados.html"), index=False)

    return outliers

# 5. Detección de outliers multivariados usando Mahalanobis
def detectar_outliers_mahalanobis(
    df_num: pd.DataFrame,
    umbral: float = 0.99,
    ruta_salida: str = None
) -> pd.Series:
    """
    Detecta outliers multivariados usando la distancia de Mahalanobis.
    Guarda gráfico de distancias si se proporciona ruta_salida.

    Args:
        df_num (pd.DataFrame): DataFrame numérico
        umbral (float): Umbral de significancia (default: 0.99)
        ruta_salida (str): Carpeta para guardar gráficos (opcional)

    Returns:
        pd.Series: Serie booleana con outliers detectados
    """
    x = df_num.dropna().values
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    dist_maha = np.array([
        np.dot(np.dot((row - mu), inv_cov), (row - mu).T)
        for row in x
    ])

    chi2_limite = chi2.ppf(umbral, df_num.shape[1])
    outliers = dist_maha > chi2_limite

    logging.info(f"Outliers multivariados detectados: {np.sum(outliers)}")

    # Guardar gráficos si se especifica la ruta
    if ruta_salida:
        Path(ruta_salida).mkdir(parents=True, exist_ok=True)
        df_dist = pd.DataFrame({"dist_mahalanobis": dist_maha})

        # Histograma
        plt.figure()
        sns.histplot(df_dist["dist_mahalanobis"], kde=True)
        plt.axvline(chi2_limite, color='r', linestyle='--', label=f'Umbral {umbral}')
        plt.title("Distribución de distancias de Mahalanobis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_salida, "mahalanobis_dist_hist.png"))
        plt.close()

        # Boxplot
        plt.figure()
        sns.boxplot(x=df_dist["dist_mahalanobis"])
        plt.title("Boxplot de distancias de Mahalanobis")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_salida, "mahalanobis_dist_boxplot.png"))
        plt.close()

    return pd.Series(outliers, index=df_num.dropna().index)

# 6. Escalamiento de variables numéricas
def escalar_variables(df_num: pd.DataFrame) -> pd.DataFrame:
    """
       Aplica escalamiento estándar (media 0, desviación 1) a variables numéricas.

       Parámetros:
           df_num (pd.DataFrame): DataFrame con variables numéricas.

       Retorna:
           pd.DataFrame: DataFrame con variables escaladas usando StandardScaler.
       """
    scaler = StandardScaler()
    df_escalado = pd.DataFrame(scaler.fit_transform(df_num),
                               columns=df_num.columns,
                               index=df_num.index
                               )
    logging.info("Escalamiento aplicado a variables numéricas")
    return df_escalado


# 7. Dumificación de variables categóricas
def dumificar_variables(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
        Aplica codificación one-hot a variables categóricas.

        Parámetros:
            df_cat (pd.DataFrame): DataFrame con variables categóricas.

        Retorna:
            pd.DataFrame: DataFrame con variables dummificadas (drop_first=True).
        """
    df_dummies = pd.get_dummies(df_cat, drop_first=True)
    logging.info("Dumificación completada")
    return df_dummies
