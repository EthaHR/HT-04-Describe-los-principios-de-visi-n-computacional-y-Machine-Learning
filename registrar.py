
import cv2
import time
from pathlib import Path

# Configuración
CARPETA_DATOS = Path("datos")
ARCHIVO_CASCADA = "haarcascade_frontalface_default.xml"
TOTAL_MUESTRAS = 50

CARPETA_DATOS.mkdir(exist_ok=True)

# Validar clasificador
cascada = cv2.CascadeClassifier(ARCHIVO_CASCADA)

if cascada.empty():
    raise FileNotFoundError("No se pudo cargar el clasificador facial")

# Abrir cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

contador = 0
ultima_captura = 0
INTERVALO = 0.35

try:
    while contador < TOTAL_MUESTRAS:
        ok, cuadro = camara.read()

        if not ok:
            print("Error al capturar imagen")
            break

        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
        rostros = cascada.detectMultiScale(
            gris, 1.2, 5, minSize=(80, 80)
        )

        mensaje = "Coloca tu rostro frente a la camara"

        if len(rostros) == 1:
            x, y, w, h = rostros[0]
            rostro = gris[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (160, 160))

            # Comprobar iluminación
            brillo = rostro.mean()

            # Comprobar nitidez
            nitidez = cv2.Laplacian(
                rostro, cv2.CV_64F
            ).var()

            if brillo < 40:
                mensaje = "Demasiado oscuro"
            elif brillo > 210:
                mensaje = "Demasiada iluminacion"
            elif nitidez < 50:
                mensaje = "Imagen borrosa"
            else:
                mensaje = "Rostro valido"

                tiempo_actual = time.time()

                if tiempo_actual - ultima_captura >= INTERVALO:
                    ruta = CARPETA_DATOS / f"usuario1_{contador:03}.png"

                    cv2.imwrite(str(ruta), rostro)
                    contador += 1
                    ultima_captura = tiempo_actual

                    mensaje = "Muestra guardada"

            cv2.rectangle(
                cuadro, (x, y), (x+w, y+h), (0, 255, 0), 2
            )

        elif len(rostros) > 1:
            mensaje = "Solo debe aparecer una persona"

        cv2.putText(
            cuadro,
            f"Muestras: {contador}/{TOTAL_MUESTRAS}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.putText(
            cuadro,
            mensaje,
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        cv2.imshow("Registro facial mejorado", cuadro)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camara.release()
    cv2.destroyAllWindows()

print(f"Registro finalizado. Muestras guardadas: {contador}")
