# Sistema de Reconocimiento Facial y Control de Acceso

Este proyecto es un sistema de reconocimiento facial desarrollado en Python utilizando OpenCV y el algoritmo LBPH (Local Binary Patterns Histograms). Permite registrar rostros de usuarios, entrenar un modelo de reconocimiento y verificar la identidad en tiempo real guardando un historial de accesos.

---

##  Requisitos Previos e Instalación

Para ejecutar este proyecto es necesario contar con Python instalado y las siguientes dependencias:

```bash
pip install opencv-contrib-python numpy
```

> **Nota:** Se requiere `opencv-contrib-python` para hacer uso del reconocedor facial LBPH.

---

##  ¿Cómo ejecutar el proyecto?

Sigue el orden de ejecución indicado a continuación para el correcto funcionamiento del sistema:

### 1. Registrar el rostro
Captura las imágenes de entrenamiento para un nuevo usuario:
```bash
python registrar.py
```
* **Resultado:** Se generarán 50 muestras de imágenes faciales dentro de la carpeta `datos/`.

### 2. Entrenar el modelo
Procesa las imágenes recopiladas y entrena el algoritmo de reconocimiento facial:
```bash
python entrenar.py
```
* **Resultado:** Genera el archivo entrenado `modelo/lbph.yml`.

### 3. Verificar la identidad
Inicia la cámara para realizar la detección y validación de usuarios en tiempo real:
```bash
python verificar.py
```
* **Resultado:** Inicia el flujo de reconocimiento en vivo y registra automáticamente los intentos de acceso en el archivo `registro_accesos.csv`.

---

##  Estructura del Proyecto

Al finalizar la ejecución de todas las etapas, el árbol de archivos y directorios del proyecto quedará de la siguiente manera:

```text
PROYECTO/
│
├── registrar.py
├── entrenar.py
├── verificar.py
├── haarcascade_frontalface_default.xml
│
├── datos/
│   ├── usuario1_000.png
│   ├── usuario1_001.png
│   └── ...
│
├── modelo/
│   └── lbph.yml
│
└── registro_accesos.csv
```
