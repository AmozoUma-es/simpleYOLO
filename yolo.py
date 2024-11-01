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

def main(task):
    # Cargar el modelo YOLOv8 de segmentación preentrenado
    model = YOLO('yolov8n-seg.pt')  # Asegúrate de que sea el modelo de segmentación

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
                    
                    # Aplicar transparencia a la máscara
                    overlay = frame.copy()
                    overlay[mask_boolean] = np.array(color, dtype=np.uint8)
                    alpha = 0.4  # Nivel de transparencia
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

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
    parser = argparse.ArgumentParser(description="Selecciona una tarea: clasificar, detectar o segmentar.")
    parser.add_argument('task', choices=['clasificar', 'detectar', 'segmentar'], help="Tarea a realizar")
    args = parser.parse_args()

    main(args.task)
