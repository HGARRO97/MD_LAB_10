import os
import subprocess
import shutil
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def eliminar_caches():
    logging.info("Eliminando carpetas '__pycache__'...")
    for root, dirs, _ in os.walk("."):
        for dir in dirs:
            if dir == "__pycache__":
                path = os.path.join(root, dir)
                shutil.rmtree(path, ignore_errors=True)
                logging.info(f"Eliminado: {path}")

    logging.info("Eliminando archivos '.pyc'...")
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc"):
                path = os.path.join(root, file)
                os.remove(path)
                logging.info(f"Eliminado: {path}")

def reinstalar_paquetes():
    logging.info("Desinstalando pandas y numpy...")
    subprocess.run(["pip", "uninstall", "pandas", "-y"], check=True)
    subprocess.run(["pip", "uninstall", "numpy", "-y"], check=True)

    logging.info("Instalando numpy==1.21.6 y pandas==1.5.3...")
    subprocess.run(["pip", "install", "numpy==1.21.6"], check=True)
    subprocess.run(["pip", "install", "pandas==1.5.3"], check=True)

if __name__ == "__main__":
    logging.info("Inicio del proceso de limpieza del entorno")
    eliminar_caches()
    reinstalar_paquetes()
    logging.info("Proceso completado. Paquetes reinstalados y cachés eliminados.")
