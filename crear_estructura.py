import os
import logging

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Ruta base del proyecto
base_dir = r"C:\Proyectos_Pycharm\Laboratorio_10_Clustering"

# Estructura del proyecto
estructura = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/preprocessing",
    "src/clustering",
    "src/evaluation",
    "src/utils",
    "tests",
    "reports",
    "references",
    "outputs"
]

# Crear carpetas
for carpeta in estructura:
    ruta = os.path.join(base_dir, carpeta)
    os.makedirs(ruta, exist_ok=True)
    logging.info(f"Directorio creado o ya existente: {ruta}")

logging.info(f"Estructura del proyecto creada en: {base_dir}")
