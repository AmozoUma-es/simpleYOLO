# simpleYOLO Python Script

Este script utiliza el modelo YOLO (You Only Look Once) para realizar diferentes tareas de procesamiento de imágenes
usando la cámara conectada al equipo. Las tareas incluyen clasificación de imágenes, detección de objetos, 
segmentación de objetos y detección de poses. La inversión de la salida de video se aplica por defecto para 
simular un efecto de espejo, pero puede desactivarse con un argumento opcional.

## Acerca de YOLO

YOLO (You Only Look Once) es una técnica de vanguardia en la detección de objetos en tiempo real, que permite la 
identificación rápida y precisa de múltiples objetos en una imagen o en un video. A diferencia de otros enfoques, YOLO 
procesa toda la imagen en una sola pasada (de ahí su nombre), lo que lo convierte en una excelente opción para aplicaciones 
de tiempo real como cámaras de vigilancia, sistemas de conducción autónoma y dispositivos móviles.

## Requisitos

- Python 3.7 o superior
- OpenCV
- NumPy
- ultralytics (para el modelo YOLO)

Instala las dependencias necesarias ejecutando:

```bash
pip install requirements.txt
```

## Uso

El script permite realizar varias tareas de procesamiento de imágenes a través de la línea de comandos, especificando 
la tarea deseada como argumento.

```bash
python yolo.py <tarea> [--no-flip]
```

### Argumentos

- `<tarea>`: La tarea que deseas ejecutar. Las opciones disponibles son:
  - `clasificar`: Realiza una clasificación de la imagen completa, mostrando la etiqueta con mayor probabilidad en el video.
  - `detectar`: Detecta objetos en el video y dibuja bounding boxes alrededor de ellos.
  - `segmentar`: Segmenta los objetos detectados y aplica colores únicos semi-transparentes sobre ellos.
  - `pose`: Detecta las poses humanas y muestra puntos clave conectados por líneas para indicar la estructura de la pose.

- `--no-flip`: Desactiva la inversión del video (efecto espejo) que se aplica por defecto.

## Tareas disponibles

### 1. Clasificar

**Descripción**: La tarea de clasificación analiza la imagen completa y determina la etiqueta que mejor describe el contenido 
principal de la imagen. Esta tarea es ideal para reconocer el tipo de escena o el objeto principal en la imagen.

**Objetivo**: Proporcionar una etiqueta representativa de la escena en tiempo real.

**Uso**:

```bash
python yolo.py clasificar
```

### 2. Detectar

**Descripción**: La tarea de detección identifica y localiza múltiples objetos dentro de la imagen. Cada objeto se muestra 
con una bounding box y su etiqueta respectiva.

**Objetivo**: Localizar y etiquetar objetos en tiempo real dentro del marco de la cámara.

![Detección de objetos, fuente: ultralytics](https://private-user-images.githubusercontent.com/47229274/382273209-af33b1c1-799c-4439-bf31-f018c8abba78.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA0NjI5ODEsIm5iZiI6MTczMDQ2MjY4MSwicGF0aCI6Ii80NzIyOTI3NC8zODIyNzMyMDktYWYzM2IxYzEtNzk5Yy00NDM5LWJmMzEtZjAxOGM4YWJiYTc4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTAxVDEyMDQ0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIwOTJiZjY4MGE0NGJlMDZkNWQ0ZTU4MjkyMjE2MjY2YjRjMTgzNDFmODQ0NWViNDkzZWExODcxNDhlOWUxZWYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ADZhN4WpdPZnDe-rYHej3Hp04kpKZUBZJBck39BV04g)

**Uso**:

```bash
python yolo.py detectar
```

### 3. Segmentar

**Descripción**: La segmentación detecta los objetos en la imagen y aplica una máscara semi-transparente sobre cada uno con 
un color único. Esto permite una visualización clara de las diferentes áreas de la imagen que contienen objetos específicos.

**Objetivo**: Ofrecer una representación visual de los objetos mediante segmentación para diferenciar áreas de la imagen en tiempo real.

![Detección de objetos, fuente: ultralytics](https://private-user-images.githubusercontent.com/47229274/382273795-b98b0600-6169-4e15-8b8f-1b42efb19722.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA0NjI5ODEsIm5iZiI6MTczMDQ2MjY4MSwicGF0aCI6Ii80NzIyOTI3NC8zODIyNzM3OTUtYjk4YjA2MDAtNjE2OS00ZTE1LThiOGYtMWI0MmVmYjE5NzIyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTAxVDEyMDQ0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQyODRmNWU1NTgwMjRmZDJiYTBkOTRiOTM5N2JmZTNhOTBlMjcwMGIzMzZiMDdiYjYyZWFkNTg0MWMzMjBjOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AEbSOxIU8NzYn3rxABJaiewPaTrqV6OUw4AVpaw5hpY)

**Uso**:

```bash
python yolo.py segmentar
```

### 4. Pose

**Descripción**: La tarea de detección de pose identifica puntos clave en el cuerpo humano, como cabeza, hombros, codos, y 
articulaciones, y los conecta mediante líneas para formar el esqueleto de la pose.

**Objetivo**: Analizar y mostrar la postura de las personas detectadas en tiempo real.

![Detección de objetos, fuente: ultralytics](https://private-user-images.githubusercontent.com/47229274/382274007-cca255d3-a90c-4340-9d61-03c0886b5074.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzA0NjI5ODEsIm5iZiI6MTczMDQ2MjY4MSwicGF0aCI6Ii80NzIyOTI3NC8zODIyNzQwMDctY2NhMjU1ZDMtYTkwYy00MzQwLTlkNjEtMDNjMDg4NmI1MDc0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTAxVDEyMDQ0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM0ZmZiMTkwYWFhZDZhMWFmY2U5MWNkOGYyNWJiMDFlMzliODc4NTU5NDFhOTJhYjBjZTkwZTI0MDI1NjBhMGImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.iAxWW7jt1bpWVu8FyGZhA4f83aGlLJx39E7UneQ53Z4)

**Uso**:

```bash
python yolo.py pose
```

## Ejemplo con `--no-flip`

Para ejecutar cualquier tarea sin invertir el video, añade el argumento `--no-flip`:

```bash
python yolo.py pose --no-flip
```
