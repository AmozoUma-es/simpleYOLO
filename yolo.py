import argparse
import cv2
import numpy as np
import random
import sys
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Variables globales para almacenar los resultados anteriores
previous_labels = []
previous_confidences = []

# Función para mostrar resultados en la consola sin llenarla de mensajes continuos
def display_results_shell(labels, confidences):
    global previous_labels, previous_confidences
    
    # Limpiar las líneas anteriores
    sys.stdout.write("\033[F" * len(previous_labels))  # Mueve el cursor hacia arriba
    sys.stdout.write("\033[K" * len(previous_labels))  # Borra las líneas anteriores

    # Mostrar solo si hay cambios en etiquetas o confidencias
    if labels != previous_labels or confidences != previous_confidences:
        for label, confidence in zip(labels, confidences):
            line = f"{label}: {int(confidence * 100)}%"
            print(line.ljust(20))  # Sobrescribe cada línea con el nuevo porcentaje
        sys.stdout.flush()
        
        # Actualizar las variables anteriores
        previous_labels = labels
        previous_confidences = confidences

# Función para dibujar la pose con colores específicos para cada par de puntos clave y una conexión lógica
def draw_pose(frame, keypoints):
    # Definición de las conexiones del esqueleto en un orden lógico según el orden proporcionado
    skeleton = [
        (3, 1), (1, 0), (0, 2), (2, 4),  # Conexiones de la cabeza: nariz -> ojo derecho -> ojo izquierdo -> oreja derecha -> oreja izquierda
        (5, 6),  # hombro derecho -> hombro izquierdo
        (5, 7), (7, 9),  # Hombro derecho -> codo derecho -> muñeca derecha
        (6, 8), (8, 10), # Hombro izquierdo -> codo izquierdo -> muñeca izquierda
        (5, 11), (6, 12),  # Hombro derecho -> cadera derecha, hombro izquierdo -> cadera izquierda
        (11, 13), (13, 15),  # Cadera derecha -> rodilla derecha -> tobillo derecho
        (12, 14), (14, 16)   # Cadera izquierda -> rodilla izquierda -> tobillo izquierdo
    ]

    # Colores para cada tipo de punto clave
    colors = {
        "nose": (0, 255, 0),           # Nariz (verde)
        "eyes": (255, 0, 0),           # Ojos (rojo)
        "ears": (0, 128, 255),         # Orejas (naranja)
        "shoulders": (255, 0, 255),    # Hombros (magenta)
        "elbows": (128, 0, 128),       # Codos (púrpura)
        "wrists": (0, 255, 255),       # Muñecas (cian)
        "hips": (255, 128, 0),         # Caderas (naranja oscuro)
        "knees": (0, 128, 128),        # Rodillas (verde oliva)
        "ankles": (128, 128, 0)        # Tobillos (oliva)
    }

    # Asignación de colores según el índice del punto clave en el modelo
    point_colors = [
        colors["nose"],       # 0 - Nariz
        colors["eyes"],       # 1 - Ojo derecho
        colors["eyes"],       # 2 - Ojo izquierdo
        colors["ears"],       # 3 - Oreja derecha
        colors["ears"],       # 4 - Oreja izquierda
        colors["shoulders"],  # 5 - Hombro derecho
        colors["shoulders"],  # 6 - Hombro izquierdo
        colors["elbows"],     # 7 - Codo derecho
        colors["elbows"],     # 8 - Codo izquierdo
        colors["wrists"],     # 9 - Muñeca derecha
        colors["wrists"],     # 10 - Muñeca izquierda
        colors["hips"],       # 11 - Cadera derecha
        colors["hips"],       # 12 - Cadera izquierda
        colors["knees"],      # 13 - Rodilla derecha
        colors["knees"],      # 14 - Rodilla izquierda
        colors["ankles"],     # 15 - Tobillo derecho
        colors["ankles"]      # 16 - Tobillo izquierdo
    ]

    # Color para las líneas de conexión (azul oscuro)
    line_color = (100, 100, 255)

    # Dibujar keypoints con los colores asignados y mostrar información en la consola
    for i in range(keypoints.shape[0]):
        x, y = keypoints[i][:2]  # Tomar solo x y y
        # Filtrar puntos no válidos (0,0) o fuera de la imagen
        if x > 0 and y > 0 and x < frame.shape[1] and y < frame.shape[0]:
            color = point_colors[i]  # Obtener el color específico para este punto
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    
    # Dibujar líneas entre keypoints para formar el esqueleto
    for i, j in skeleton:
        x1, y1 = keypoints[i][:2]
        x2, y2 = keypoints[j][:2]
        # Filtrar conexiones con puntos no válidos
        if (x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0 and 
            x1 < frame.shape[1] and y1 < frame.shape[0] and 
            x2 < frame.shape[1] and y2 < frame.shape[0]):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 2)

def main(task, flip_video):
    # Cargar el modelo YOLOv8 de segmentación o de pose, según la tarea
    if task == 'pose':
        model = YOLO('yolo11n-pose.pt')  # Cargar el modelo de detección de pose
    else:
        model = YOLO('yolo11n-seg.pt')  # Cargar el modelo de segmentación

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    # Diccionario para almacenar colores únicos para cada clase
    colors = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir frame.")
            break

        # Invertir el frame si flip_video es True
        if flip_video:
            frame = cv2.flip(frame, 1)

        results = model(frame)

        # Listas para etiquetas y confianzas
        labels = []
        confidences = []
        
        if task == 'clasificar':
            # Tomar la clase de mayor confianza y mostrarla como clasificación
            label = results[0].names[int(results[0].boxes[0].cls[0])]
            confidence = results[0].boxes[0].conf[0]
            labels.append(label)
            confidences.append(confidence)
            cv2.putText(frame, f'{label}: {int(confidence * 100)}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif task == 'detectar' or task == 'segmentar':
            # Asignar colores únicos a cada clase detectada si no están ya en el diccionario
            for box in results[0].boxes:
                cls = int(box.cls[0])
                confidence = box.conf[0]
                if cls not in colors:
                    colors[cls] = [random.randint(0, 255) for _ in range(3)]  # Color aleatorio para cada clase
                
                color = colors[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results[0].names[cls]

                # Añadir label y confidence a las listas
                labels.append(label)
                confidences.append(confidence)
                
                # Dibujar bounding box y etiqueta con confianza en el video
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label}: {int(confidence * 100)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Solo ejecutar segmentación si la tarea es 'segmentar'
            if task == 'segmentar' and results[0].masks:
                for mask, box in zip(results[0].masks.data, results[0].boxes):
                    cls = int(box.cls[0])
                    color = colors[cls]  # Usar el mismo color de la clase para la máscara
                    
                    mask_np = mask.cpu().numpy()  # Convertir a NumPy
                    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                    mask_boolean = mask_resized.astype(bool)
                    
                    # Aplicar transparencia a la región de la máscara
                    overlay = frame.copy()
                    overlay[mask_boolean] = np.array(color, dtype=np.uint8)
                    alpha = 0.4  # Nivel de transparencia
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        elif task == 'pose':
            # Detección de poses
            for pose in results[0].keypoints.data:
                keypoints = pose.cpu().numpy()  # Convertir los keypoints a NumPy
                draw_pose(frame, keypoints)

        # Llamada a la función para actualizar resultados en la consola
        display_results_shell(labels, confidences)

        # Mostrar el frame resultante
        cv2.imshow("Resultados YOLO", frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Selecciona una tarea: clasificar, detectar, segmentar o pose.")
    parser.add_argument('task', choices=['clasificar', 'detectar', 'segmentar', 'pose'], help="Tarea a realizar")
    parser.add_argument('--no-flip', action='store_false', dest='flip_video', help="No invertir el video como un espejo")
    args = parser.parse_args()

    main(args.task, args.flip_video)
