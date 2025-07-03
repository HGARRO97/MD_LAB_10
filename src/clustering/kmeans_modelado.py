import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import os

# 1. Método del codo
def plot_elbow_method(df: pd.DataFrame, max_k: int = 10, save_path: str = None) -> None:
    """
    Grafica el método del codo para identificar el número óptimo de clusters K.

    Args:
        df (pd.DataFrame): DataFrame con variables numéricas preparadas para clustering.
        max_k (int): Número máximo de clusters a evaluar. Default es 10.
        save_path (str): Ruta donde se guardará el gráfico del codo (opcional).

    Returns:
        None
    """
    inertia = []
    for k in range(1, max_k + 1):
        modelo = KMeans(n_clusters=k, random_state=42)
        modelo.fit(df)
        inertia.append(modelo.inertia_)
        logging.info(f"K={k}, Inercia={modelo.inertia_:.2f}")

    plt.figure()
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Inercia")
    plt.title("Método del Codo")
    plt.grid(True)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"Gráfico del codo guardado en: {save_path}")
    plt.close()

# 2. Entrenamiento de K-means
def entrenar_kmeans(df: pd.DataFrame, k: int) -> KMeans:
    """
    Entrena un modelo K-means con el número de clusters especificado.

    Args:
        df (pd.DataFrame): DataFrame numérico escalado para clustering.
        k (int): Número de clusters.

    Returns:
        KMeans: Modelo KMeans entrenado.
    """
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(df)
    logging.info(f"K-means entrenado con K={k}")
    return modelo

# 3. Asignación de clusters
def asignar_clusters(df: pd.DataFrame, modelo: KMeans) -> pd.DataFrame:
    """
    Asigna las etiquetas de cluster generadas por K-means al DataFrame.

    Args:
        df (pd.DataFrame): DataFrame original (numérico).
        modelo (KMeans): Modelo entrenado de K-means.

    Returns:
        pd.DataFrame: DataFrame con nueva columna "cluster".
    """
    df_resultado = df.copy()
    df_resultado["cluster"] = modelo.predict(df)
    logging.info("Clusters asignados al DataFrame")
    return df_resultado

# 4. Análisis por cluster
def resumen_clusters(df: pd.DataFrame, save_dir: str, var_x: str, var_y: str) -> pd.DataFrame:
    """
    Genera un resumen estadístico por cluster y visualizaciones asociadas.

    Args:
        df (pd.DataFrame): DataFrame con columna 'cluster' asignada.
        save_dir (str): Carpeta de salida para guardar imágenes y resumen.
        var_x (str): Nombre de la variable para eje X en scatterplot.
        var_y (str): Nombre de la variable para eje Y en scatterplot.

    Returns:
        pd.DataFrame: Resumen promedio por cluster (centroides).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    resumen = df.groupby("cluster").mean(numeric_only=True).round(4)
    resumen.to_html(os.path.join(save_dir, "resumen_clusters.html"))

    # Gráfico scatter 2D con variables especificadas
    plt.figure()
    sns.scatterplot(
        data=df,
        x=var_x,
        y=var_y,
        hue="cluster",
        palette="tab10"
    )
    plt.title(f"Distribución de clusters: {var_x} vs {var_y}")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_clusters.png"), bbox_inches="tight")
    plt.close()

    # Gráfico de barras con porcentaje por cluster
    plt.figure()
    total = len(df)
    counts = df["cluster"].value_counts().sort_index()
    porcentajes = (counts / total * 100).round(2)
    ax = counts.plot(kind="bar")
    for i, p in enumerate(porcentajes):
        ax.text(i, counts.iloc[i] + total * 0.01, f"{p}%", ha="center")
    plt.title("Cantidad de observaciones por cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_counts.png"))
    plt.close()

    logging.info("Resumen estadístico por cluster generado")
    return resumen


