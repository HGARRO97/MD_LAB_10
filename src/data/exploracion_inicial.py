from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def diagnostico_inicial(df: pd.DataFrame, save_dir: str = "outputs/01_diagnostico") -> None:
    """
    Realiza un análisis exploratorio inicial del DataFrame.
    Genera gráficos, tabla de nulos, y resumen estadístico.

    Args:
        df (pd.DataFrame): Datos a analizar.
        save_dir (str): Carpeta donde guardar los gráficos y reportes.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Resumen estructural
    logging.info("Resumen general del dataset:")
    logging.info(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    logging.info(f"Columnas:\n{df.dtypes}")
    logging.info("Resumen estadístico:\n" + str(df.describe().round(2)))

    # Nulos con tipo de dato
    nulos = df.isnull().sum()
    total = len(df)
    porcentaje = ((nulos / total) * 100).round(2)
    tipos = df.dtypes
    df_nulos = pd.DataFrame({
        "nulos": nulos,
        "porcentaje": porcentaje,
        "tipo_dato": tipos
    }).sort_values("nulos", ascending=False)

    logging.info("Conteo de nulos por columna:\n" + str(df_nulos))

    # Exportar diagnósticos
    df_nulos.to_excel(os.path.join(save_dir, "diagnostico_nulos.xlsx"))
    df.describe().round(2).to_html(os.path.join(save_dir, "resumen_estadistico.html"))

    # Gráficos: histogramas y boxplots
    for col in df.select_dtypes(include="number"):
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribución: {col}")
        plt.savefig(os.path.join(save_dir, f"hist_{col}.png"), bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.savefig(os.path.join(save_dir, f"box_{col}.png"), bbox_inches="tight")
        plt.close()

    # Heatmap
    if df.select_dtypes(include="number").shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "correlation_matrix.png"))
        plt.close()

    logging.info(f"Gráficos y reportes guardados en: {save_dir}")



