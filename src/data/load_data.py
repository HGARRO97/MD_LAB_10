import pandas as pd
import logging
import os

def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.

    Args:
        ruta_csv (str): Ruta del archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        if not os.path.exists(ruta_csv):
            logging.error(f"No se encontró el archivo en la ruta: {ruta_csv}")
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")

        df = pd.read_csv(ruta_csv, encoding="utf-8")
        logging.info(f"Datos cargados correctamente desde: {ruta_csv}")
        return df

    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
        raise

