
import cv2
import csv
import time
from pathlib import Path
from datetime import datetime

# Configuracion
ARCHIVO_MODELO = Path("modelo/lbph.yml")
ARCHIVO_CASCADA = Path("haarcascade_frontalface_default.xml")
ARCHIVO_LOG = Path("registro_accesos.csv")

UMBRAL = 65
MAX_INTENTOS = 5
TIEMPO_BLOQUEO = 10
INTERVALO_REGISTRO = 1.5

# Validar archivos
if not ARCHIVO_MODELO.exists():
    raise FileNotFoundError("No existe el modelo entrenado")

if not ARCHIVO_CASCADA.exists():
    raise FileNotFoundError("No existe el clasificador facial")

if not hasattr(cv2, "face"):
    raise RuntimeError("Se necesita opencv-contrib-python")

# Cargar modelo
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.read(str(ARCHIVO_MODELO))

# Cargar clasificador
cascada = cv2.CascadeClassifier(str(ARCHIVO_CASCADA))

if cascada.empty():
    raise RuntimeError("No se pudo cargar el clasificador")

# Crear archivo de registros
if not ARCHIVO_LOG.exists():
    with open(ARCHIVO_LOG, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["Fecha", "Hora", "Resultado", "Distancia"])

# Funcion para registrar accesos
def registrar_evento(resultado, distancia=""):
    ahora = datetime.now()

    with open(ARCHIVO_LOG, "a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            ahora.strftime("%Y-%m-%d"),
            ahora.strftime("%H:%M:%S"),
            resultado,
            distancia
        ])

# Abrir camara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la camara")

intentos_fallidos = 0
bloqueado_hasta = 0
ultimo_intento = 0
ultimo_resultado = None

try:
    while True:
        ok, cuadro = camara.read()

        if not ok:
            break

        gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)

        rostros = cascada.detectMultiScale(
            gris, 1.2, 5, minSize=(80, 80)
        )

        mensaje = "ACCESO DENEGADO"
        tiempo_actual = time.time()

        # Comprobar bloqueo
        if tiempo_actual < bloqueado_hasta:
            restante = int(bloqueado_hasta - tiempo_actual) + 1
            mensaje = f"SISTEMA BLOQUEADO: {restante}s"

        elif len(rostros) == 1:
            x, y, w, h = rostros[0]

            rostro = gris[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (160, 160))

            etiqueta, distancia = modelo.predict(rostro)

            autorizado = (
                etiqueta == 1 and distancia < UMBRAL
            )

            # Registrar como maximo un intento cada 1.5 segundos
            if tiempo_actual - ultimo_intento >= INTERVALO_REGISTRO:

                ultimo_intento = tiempo_actual

                if autorizado:
                    mensaje = "ACCESO SIMULADO AUTORIZADO"
                    intentos_fallidos = 0
                    registrar_evento(
                        "AUTORIZADO", round(float(distancia), 2)
                    )

                else:
                    intentos_fallidos += 1
                    mensaje = "ACCESO DENEGADO"

                    registrar_evento(
                        "DENEGADO", round(float(distancia), 2)
                    )

                    if intentos_fallidos >= MAX_INTENTOS:
                        bloqueado_hasta = (
                            tiempo_actual + TIEMPO_BLOQUEO
                        )
                        intentos_fallidos = 0
                        mensaje = "BLOQUEO TEMPORAL"

            elif ultimo_resultado is not None:
                mensaje = ultimo_resultado

            ultimo_resultado = mensaje

            cv2.rectangle(
                cuadro, (x, y), (x+w, y+h), (0, 220, 0), 2
            )

            cv2.putText(
                cuadro,
                f"Distancia: {distancia:.1f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

        elif len(rostros) > 1:
            mensaje = "Solo debe aparecer una persona"

        # Mostrar resultado
        cv2.putText(
            cuadro,
            mensaje,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        cv2.putText(
            cuadro,
            f"Intentos fallidos: {intentos_fallidos}/{MAX_INTENTOS}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )

        cv2.imshow("Sistema de verificacion mejorado", cuadro)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camara.release()
    cv2.destroyAllWindows()
