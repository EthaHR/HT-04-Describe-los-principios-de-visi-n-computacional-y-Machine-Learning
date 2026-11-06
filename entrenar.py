import cv2
import numpy as np
from pathlib import Path

# Buscar y cargar las imágenes de entrenamiento
rutas = sorted(Path("datos").glob("usuario1_*.png"))
imagenes = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE) for r in rutas]
imagenes = [im for im in imagenes if im is not None]

if len(imagenes) < 15:
    raise RuntimeError("Registra al menos 15 muestras válidas")

# Crear y entrenar el modelo LBPH
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.train(imagenes, np.full(len(imagenes), 1, dtype=np.int32))

# Guardar el modelo entrenado
Path("modelo").mkdir(exist_ok=True)
modelo.write("modelo/lbph.yml")

print(f"Entrenamiento finalizado; imágenes: {len(imagenes)}")