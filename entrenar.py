
import cv2
import numpy as np
from pathlib import Path

# Rutas del proyecto
CARPETA_DATOS = Path("datos")
CARPETA_MODELO = Path("modelo")
ARCHIVO_MODELO = CARPETA_MODELO / "lbph.yml"

# Comprobar carpeta de datos
if not CARPETA_DATOS.exists():
    raise FileNotFoundError("No existe la carpeta datos")

# Buscar imágenes
rutas = sorted(CARPETA_DATOS.glob("usuario1_*.png"))

if not rutas:
    raise RuntimeError("No se encontraron imagenes de entrenamiento")

imagenes = []

for ruta in rutas:
    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        print(f"Imagen no valida: {ruta.name}")
        continue

    if imagen.shape != (160, 160):
        print(f"Tamano incorrecto: {ruta.name}")
        continue

    imagenes.append(imagen)

# Validar cantidad
if len(imagenes) < 20:
    raise RuntimeError(
        f"Se necesitan al menos 20 imagenes validas. "
        f"Encontradas: {len(imagenes)}"
    )

# Comprobar disponibilidad de LBPH
if not hasattr(cv2, "face"):
    raise RuntimeError(
        "Falta cv2.face. Instala opencv-contrib-python"
    )

# Crear y entrenar modelo
modelo = cv2.face.LBPHFaceRecognizer_create()

etiquetas = np.full(
    len(imagenes), 1, dtype=np.int32
)

modelo.train(imagenes, etiquetas)

# Guardar modelo
CARPETA_MODELO.mkdir(exist_ok=True)
modelo.write(str(ARCHIVO_MODELO))

# Verificar guardado
if not ARCHIVO_MODELO.exists():
    raise RuntimeError("No se pudo guardar el modelo")

print("Entrenamiento finalizado correctamente")
print(f"Imagenes utilizadas: {len(imagenes)}")
print(f"Modelo guardado en: {ARCHIVO_MODELO}")
