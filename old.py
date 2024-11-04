import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os
import matplotlib.pyplot as plt
class VideoProcessor:
    def __init__(self, modelo='yolov8n.pt', source='camera', video_path=None, ruta_carpeta=None, save = False, n = 1,depurar = False):
        # Inicialización de modelo y dispositivo
        self.yolo_model = YOLO(modelo)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.yolo_model.to(self.device)
        self.modelo_pelota = YOLO('yolov8x.pt')
        
        # Configuración de fuente de video
        self.source = source
        self.n = n
        self.depurar = depurar
        if source == 'camera':
            self.cap = cv2.VideoCapture(1)
        elif source == 'video' and video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            raise ValueError("Invalid source or missing video path")
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open {'camera' if source == 'camera' else 'video file'}.")
        else:
            print(f"{'Camera' if source == 'camera' else 'Video file'} opened successfully.")
        
        # Colores predefinidos para detección
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')
        
        # Configuración de directorio de guardado
        self.ruta_carpeta = ruta_carpeta
        if ruta_carpeta:
            os.makedirs(self.ruta_carpeta, exist_ok=True)
        self.save = save
        
        # Atributos para cálculo de posesión del balón
        self.frames_equipo_1 = 0
        self.frames_equipo_2 = 0
        self.total_frames = 0

        
    def procesar(self, num_frames=None, frame_skip=0):
        frames_leidos = 0
        frames_procesados = 0
        total_time = 0
        while self.cap.isOpened():
            if num_frames is not None and frames_procesados >= num_frames:
                break

            for _ in range(frame_skip + 1):
                ret, frame = self.cap.read()
                if not ret:
                    print("Fin del video o error de lectura.")
                    return
                frames_leidos += 1

            start_time = time.time()

            # Factor de escala
            self.n = 1
            original_height, original_width = frame.shape[:2]
            frame = cv2.resize(frame, (int(original_width * self.n), int(original_height * self.n)))

            # Detectar solo personas con batch processing
            results = self.yolo_model(frame, stream=True,classes = [0,32] ,conf=0.5)  # 0 es el índice para 'person'

            # Procesar resultados
            for result in results:
                detections = result.boxes
                print(f"Frame {frames_leidos} detecciones: {len(detections)}")
                frame_with_detections = self.process_detections(frame, detections)
                
                # Mostrar el frame con las detecciones
                cv2.imshow('Detecciones', frame_with_detections)
                if self.save == True:
                    self.guardar_frame(frame_with_detections,frames_leidos)

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            print(f"Tiempo de procesamiento del frame {frames_leidos}: {elapsed_time:.4f} segundos")

            frames_procesados += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        avg_time = total_time / frames_procesados if frames_procesados > 0 else 0
        print(f"Frames procesados: {frames_procesados}")
        print(f"Tiempo promedio de procesamiento por frame: {avg_time:.4f} segundos")
        print(f"FPS promedio: {1/avg_time:.2f}")

        self.cap.release()
        cv2.destroyAllWindows()
        print("Captura de video liberada.")
        
    def modify_frame(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara para colores azules
        mask_blue = cv2.inRange(hsv_frame, np.array([75, 50, 50]), np.array([135, 255, 255]))
        
        # Suavizar y limpiar la máscara (operaciones morfológicas)
        #kernel = np.ones((5, 5), np.uint8)
        #mask_blue = cv2.GaussianBlur(mask_blue, (5, 5), 0)
        #mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        #mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        
        # Reducir saturación en áreas azules
        hsv_frame[:, :, 1] = np.where(mask_blue > 0, 0, hsv_frame[:, :, 1])
        
        # Ajustar HSV perfil rojo amarillo
        hsv_frame[:, :, 0] = np.clip(hsv_frame[:, :, 0], 37, 179)
        hsv_frame[:, :, 2] = 255  # Maximizar el valor (brillo)
        #Ajustar HSV perfil blancos
        #hsv_frame[:, :, 0] = np.clip(hsv_frame[:, :, 0], 29, 132)  # Asegúrate de que H no exceda los límites
        #hsv_frame[:, :, 2] = 255  # Maximizar el valor (brillo)

        # ajustar vivos 

        H_low, H_high = 0, 179
        S_low, S_high = 0, 255
        V_low, V_high = 255, 255

        # Aplicar los límites a los canales H, S, y V
        #hsv_frame[:, :, 0] = np.clip(hsv_frame[:, :, 0], H_low, H_high)  # Canal H
        #hsv_frame[:, :, 1] = np.clip(hsv_frame[:, :, 1], S_low, S_high)  # Canal S
        #hsv_frame[:, :, 2] = np.clip(hsv_frame[:, :, 2], V_low, V_high)  # Canal V
        
        return cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)

    def process_detections(self, frame, detections):
        modified_frame = self.modify_frame(frame)
        player_colors = []
        
        for detection in detections:
            if len(detection.xyxy) >= 1:
                x1, y1, x2, y2 = map(int, detection.xyxy[0][:4])
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                half_height = (y2 - y1) // 2
                half_width = (x2 - x1) // 2
                centered_x1 = max(x1, mid_x - half_width // 2)
                centered_x2 = min(x2, mid_x + half_width // 2)
                centered_y1 = max(y1, mid_y - half_height // 2)
                centered_y2 = min(y2, mid_y + half_height // 2)

                player_img = modified_frame[centered_y1:centered_y2, centered_x1:centered_x2]
                avg_color_hsv = cv2.cvtColor(np.uint8([[player_img.mean(axis=(0,1))]]), cv2.COLOR_BGR2HSV)[0][0]
                player_colors.append(tuple(avg_color_hsv.astype(int)))
    

        team_colors = self.get_team_colors(player_colors)
        if self.depurar == False:
            frame_with_detections = self.draw_detections(frame.copy(), detections, team_colors)
        else:
            frame_with_detections = self.draw_detections(modified_frame, detections, team_colors)
        # Mostrar el frame con las detecciones
        cv2.imshow('Detecciones', frame_with_detections)
        
        return frame_with_detections

    def get_team_colors(self, player_colors):
        color_counts = Counter(player_colors)
        unique_colors = list(color_counts.keys())
        
        if len(unique_colors) < 2:
            return None
        
        distances = np.linalg.norm(np.array(unique_colors)[:, np.newaxis] - np.array(unique_colors), axis=2)
        i, j = np.unravel_index(distances.argmax(), distances.shape)
        
        return tuple(sorted([unique_colors[i], unique_colors[j]]))

    


    def guardar_frame(self, frame, frame_index):
            if self.ruta_carpeta is None:
                print("No se ha especificado una carpeta para guardar los frames.")
                return
            try:
                os.makedirs(self.ruta_carpeta, exist_ok=True)
                output_frame_filename = f"frame_clasificado_{frame_index}.png"
                saved_path = os.path.join(self.ruta_carpeta, output_frame_filename)
                cv2.imwrite(saved_path, frame)
                if os.path.exists(saved_path):
                    print(f'Imagen del frame guardada como {output_frame_filename}')
                else:
                    print(f'Error al guardar la imagen del frame {frame_index}.')
            except Exception as e:
                print(f"Error al guardar el frame {frame_index}: {str(e)}")

    def draw_detections(self, frame, detections, team_colors):
        for detection in detections:
            if len(detection) >= 1:
                # Extracción y procesamiento de las coordenadas del jugador
                bbox = detection.xyxy[0][:4].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                
                mid_coords = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                half_dims = np.array([(x2 - x1) // 2, (y2 - y1) // 2])
                
                # Cálculo de coordenadas para recortes de detección
                centered_coords = np.array([
                    max(x1, mid_coords[0] - half_dims[0] // 2),
                    max(y1, mid_coords[1] - half_dims[1] // 2),
                    min(x2, mid_coords[0] + half_dims[0] // 2),
                    min(y2, mid_coords[1] + half_dims[1] // 2)
                ])
                
                scale_factor = 3
                expanded_half_dims = (half_dims * scale_factor).astype(int)
                expanded_coords = np.array([
                    max(0, mid_coords[0] - expanded_half_dims[0]),
                    max(0, mid_coords[1] - expanded_half_dims[1]),
                    min(frame.shape[1], mid_coords[0] + expanded_half_dims[0]),
                    min(frame.shape[0], mid_coords[1] + expanded_half_dims[1])
                ])
                
                # Extraer región para color promedio del jugador
                player_img = frame[centered_coords[1]:centered_coords[3], 
                                centered_coords[0]:centered_coords[2]]
                avg_color = player_img.mean(axis=(0, 1))
                avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color]]), 
                                        cv2.COLOR_BGR2HSV)[0][0]
                
                # Determinar equipo basado en color promedio
                team = 1 if team_colors and np.linalg.norm(avg_color_hsv - team_colors[0]) < np.linalg.norm(avg_color_hsv - team_colors[1]) else 2
                
                # Extraer región expandida para detección de pelota
                player_img_2 = frame[expanded_coords[1]:expanded_coords[3], 
                                     expanded_coords[0]:expanded_coords[2]]
                resultados_pelota = self.modelo_pelota(player_img_2, classes=[32])
                
                # Detección de la pelota y actualización de posesión
                for resultado in resultados_pelota:
                    if resultado.boxes and len(resultado.boxes) > 0:
                        for box in resultado.boxes:
                            box_coords = box.xyxy[0].cpu().numpy().astype(int)
                            rel_coords = box_coords / np.array([
                                player_img_2.shape[1], player_img_2.shape[0],
                                player_img_2.shape[1], player_img_2.shape[0]
                            ])
                            region_dims = np.array([
                                expanded_coords[2] - expanded_coords[0],
                                expanded_coords[3] - expanded_coords[1]
                            ])
                            abs_coords = np.array([
                                expanded_coords[0] + int(rel_coords[0] * region_dims[0]),
                                expanded_coords[1] + int(rel_coords[1] * region_dims[1]),
                                expanded_coords[0] + int(rel_coords[2] * region_dims[0]),
                                expanded_coords[1] + int(rel_coords[3] * region_dims[1])
                            ])
                            cv2.rectangle(frame, (abs_coords[0], abs_coords[1]), 
                                          (abs_coords[2], abs_coords[3]), 
                                          (0, 255, 0), 1)

                        # Actualizar conteo de posesión según el equipo
                        if team == 1:
                            self.frames_equipo_1 += 1
                        elif team == 2:
                            self.frames_equipo_2 += 1
                        self.total_frames += 1

                # Dibujar jugador y equipo
                color_bgr = cv2.cvtColor(np.uint8([[avg_color_hsv]]), 
                                         cv2.COLOR_HSV2BGR)[0][0].tolist()
                cv2.rectangle(frame, (centered_coords[0], centered_coords[1]),
                              (centered_coords[2], centered_coords[3]), 
                              color_bgr, 1)
                cv2.circle(frame, (mid_coords[0], mid_coords[1]), 3, color_bgr, -1)
                cv2.putText(frame, f'Equipo {team}', 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Calcular y mostrar porcentaje de posesión
        posesion_1 = (self.frames_equipo_1 / self.total_frames) * 100 if self.total_frames > 0 else 0
        posesion_2 = (self.frames_equipo_2 / self.total_frames) * 100 if self.total_frames > 0 else 0

        # Dibujar cuadro negro para mostrar posesión
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, "----Posesión----", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        cv2.putText(frame, "|", (140, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convertir colores a tuplas de enteros en draw_possession_text
        cv2.circle(frame, (50, 60), 10, tuple(map(int, team_colors[0])), -1)
        cv2.putText(frame, f'{posesion_1:.2f}%', (70, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.circle(frame, (220, 60), 10, tuple(map(int, team_colors[1])), -1)
        cv2.putText(frame, f'{posesion_2:.2f}%', (160, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame