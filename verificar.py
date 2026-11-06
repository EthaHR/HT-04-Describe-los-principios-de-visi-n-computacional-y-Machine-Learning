import cv2

# Cargar el modelo entrenado y el clasificador en cascada local
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.read("modelo/lbph.yml")

cascada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

UMBRAL = 65

while True:
    ok, cuadro = camara.read()
    if not ok:
        break

    gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    rostros = cascada.detectMultiScale(gris, 1.2, 5, minSize=(80, 80))

    mensaje = "ACCESO DENEGADO"

    if len(rostros) == 1:
        x, y, w, h = rostros[0]
        rostro = cv2.resize(gris[y : y + h, x : x + w], (160, 160))
        etiqueta, distancia = modelo.predict(rostro)

        if etiqueta == 1 and distancia < UMBRAL:
            mensaje = "ACCESO SIMULADO AUTORIZADO"

        cv2.rectangle(cuadro, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            cuadro,
            f"Distancia: {distancia:.1f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        cuadro,
        mensaje,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Acceso local demostrativo", cuadro)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()