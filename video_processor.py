import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from collections import Counter
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
class KSoccer:
    def __init__(self, model='yolov8n.pt', source='camera', path=None, ruta_carpeta=None, save = False, scale_factor = 1,depurar = False):
        # Inicialización de model y dispositivo
        self.yolo_model = YOLO(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_pelota = YOLO(model)
        self.yolo_model.to(self.device)
        self.model_pelota.to(self.device)
        # Configuración de fuente de video
        self.source = source
        self.scale_factor = scale_factor
        self.depurar = depurar

        if source == "image":
            self.frame = cv2.imread(path)
            self.cap = None
        elif source == 'camera':
            self.cap = cv2.VideoCapture(0)
        elif source == 'video' and path:
            self.cap = cv2.VideoCapture(path)
        else:
            raise ValueError("Invalid source or missing video path")
        
        if self.cap and not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open {'camera' if source == 'camera' else 'video file'}.")
        else:
            print(f"{'Camera' if source == 'camera' else 'File'} opened successfully.")
        
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
        frames_procesados = 0
        frames_leidos = 0
        total_time = 0

        if self.source == 'camera' or self.source == 'video':
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

                frame = self.process_frame(frame)
                
                if self.save == True:
                    self.guardar_frame(frame,frames_leidos)

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
        
        elif self.source == 'image':
            frame = self.process_frame(self.frame)
            if self.save == True:
                self.guardar_frame(frame)


    def process_frame(self, frame):
        # Factor de escala

        original_height, original_width = frame.shape[:2]
        frame = cv2.resize(frame, (int(original_width * self.scale_factor), int(original_height * self.scale_factor)))

        # Detectar solo personas con batch processing
        results = self.yolo_model(frame, classes=[0], conf=0.25)
       

        ball_results = self.model_pelota(frame, classes=[32], conf = 0.1)

        

        frame_with_detections = frame.copy()
        # initialize any numpy array
        ball_bbox = np.zeros(1)

        for ball_result in ball_results:
            ball_detections = ball_result.boxes

            for detection in ball_detections:
                if len(detection.xyxy) >= 1:
                    ball_bbox = detection.xyxy[0].cpu().numpy().astype(int)
                    ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                    ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                    cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), radius = 10, color = (0,0,255), thickness = -1)

        for result in results:
            # Procesar resultados
            detections = result.boxes
            processed_frame = self.preprocess_frame(frame)
            player_detections = []
            

            # Process each detection
            for detection in detections:
                if len(detection.xyxy) >= 1:
                    bbox = detection.xyxy[0].cpu().numpy().astype(int)

                    player_info = self.process_player(processed_frame, ball_bbox, bbox)

                    if player_info:
                        player_detections.append(player_info)
            
            # Sort detections by y-coordinate for consistent processing
            player_detections.sort(key=lambda x: x['bbox'][1])
            
            # Process team assignments and ball detection
            frame_with_detections = self.draw_results(frame, player_detections)
            
            # Add debug information
            if self.depurar:
                cv2.putText(frame_with_detections, 
                            f"Players detected: {len(player_detections)}", 
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar el frame con las detecciones
        cv2.imshow('Detecciones', frame_with_detections)
        return frame_with_detections
        


    def process_player(self, frame, ball_bbox, player_bbox):
        """
        Extract player information including team colors and possible ball possession
        """
        x1, y1, x2, y2 = player_bbox
        
        # Calculate expanded region for better ball detection
        mid_coords = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
        half_dims = np.array([(x2 - x1) // 2, (y2 - y1) // 2])
        
        # Calculate expanded coordinates with scale factor
        scale_factor = 3
        expanded_half_dims = (half_dims * scale_factor).astype(int)
        expanded_coords = np.array([
            max(0, mid_coords[0] - expanded_half_dims[0]),
            max(0, mid_coords[1] - expanded_half_dims[1]),
            min(frame.shape[1], mid_coords[0] + expanded_half_dims[0]),
            min(frame.shape[0], mid_coords[1] + expanded_half_dims[1])
        ])
        
        # Extract jersey region (upper body)
        body_height = y2 - y1
        jersey_y1 = y1 + int(body_height * 0.2)
        jersey_y2 = y1 + int(body_height * 0.5)
        jersey_region = frame[jersey_y1:jersey_y2, x1:x2]
        
        if jersey_region.size == 0:
            return None
            
        # Get dominant colors using k-means
        jersey_color = self.get_dominant_color(jersey_region)
        
        # Check for ball possession using expanded region
        has_ball = self.ball_possession(ball_bbox, expanded_coords)
        
        return {
            'bbox': player_bbox,
            'jersey_color': jersey_color,
            'has_ball': has_ball,
            'expanded_coords': expanded_coords  # Add this for visualization if needed
        }

    def ball_possession(self, ball_bbox, expanded_coords):
        """
        Enhanced ball detection using expanded region around player
        """
        if ball_bbox.any():
            ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
            
            # Check if ball center is within expanded region
            if (expanded_coords[0] <= ball_center_x <= expanded_coords[2] and
                expanded_coords[1] <= ball_center_y <= expanded_coords[3]):
                
                # Calculate relative position in expanded region
                rel_x = (ball_center_x - expanded_coords[0]) / (expanded_coords[2] - expanded_coords[0])
                rel_y = (ball_center_y - expanded_coords[1]) / (expanded_coords[3] - expanded_coords[1])
                
                # Give higher weight to balls detected closer to player center
                center_distance = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
                if center_distance < 0.7:  # Adjust this threshold as needed
                    return True
        
        return False

    def get_dominant_color(self, image, k=3):
        """
        Improved dominant color extraction with better filtering
        """
        try:
            # Resize for consistency
            #image = cv2.resize(image, (50, 50))
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for valid colors (exclude very dark/light/unsaturated)
            mask = cv2.inRange(hsv_image, 
                              np.array([0, 50, 50]), 
                              np.array([180, 255, 255]))
            
            # Apply mask
            valid_pixels = hsv_image[mask > 0]
            
            if len(valid_pixels) < 10:  # Not enough valid pixels
                return None
                
            # Reshape for k-means
            valid_pixels = valid_pixels.reshape(-1, 3)
            
            # Apply k-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(valid_pixels)
            
            # Get cluster sizes
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            
            # Sort clusters by size
            sorted_indices = np.argsort(-counts)
            centers = kmeans.cluster_centers_[sorted_indices]
            
            # Return the largest cluster center that meets our criteria
            for center in centers:
                if 20 < center[2] < 235 and center[1] > 40:  # Value and Saturation thresholds
                    return center
                    
            return None
        except Exception:
            return None

    

    def assign_teams(self, player_detections):
        """
        Improved team assignment with initialization
        """
        if not hasattr(self, 'team_colors_history'):
            self.team_colors_history = []
        
        # Extract valid jersey colors
        jersey_colors = [p['jersey_color'] for p in player_detections if p['jersey_color'] is not None]
        
        if len(jersey_colors) < 2:
            return
        
        # Convert to numpy array
        colors = np.array(jersey_colors)
        
        # If we don't have established team colors yet
        if not self.team_colors_history:
            # Initial clustering
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(colors)
            self.team_colors_history.append(kmeans.cluster_centers_)
        
        # Use most recent team colors
        current_colors = self.team_colors_history[-1]
        
        # Assign teams to players
        for player in player_detections:
            if player['jersey_color'] is not None:
                dist1 = np.linalg.norm(player['jersey_color'] - current_colors[0])
                dist2 = np.linalg.norm(player['jersey_color'] - current_colors[1])
                player['team'] = 1 if dist1 < dist2 else 2

    def update_possession(self, player_detections):
        """
        Simplified possession calculation
        """
        team1_has_ball = False
        team2_has_ball = False
        
        # Check for ball possession
        for player in player_detections:
            if player.get('has_ball', False) and player.get('team') is not None:
                if player['team'] == 1:
                    team1_has_ball = True
                elif player['team'] == 2:
                    team2_has_ball = True
        
        # Update frame counts
        if team1_has_ball:
            self.frames_equipo_1 += 1
            self.total_frames += 1

        elif team2_has_ball:
            self.frames_equipo_2 += 1
            self.total_frames += 1

    
    def draw_results(self, frame, player_detections):
        """
        Enhanced visualization with expanded regions and ball possession
        """
        # Assign teams first
        self.assign_teams(player_detections)
        
        # Update possession
        self.update_possession(player_detections)
        
        # Draw player boxes and team assignments
        for player in player_detections:
            bbox = player['bbox']
            team = player.get('team')
            has_ball = player.get('has_ball', False)
            expanded_coords = player.get('expanded_coords')
            
            if team is not None:
                # Draw player box with team color
                color = (0, 255, 0) if team == 1 else (255, 0, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw expanded region if ball possession detected
                if has_ball and expanded_coords is not None:
                    cv2.rectangle(frame, 
                                (expanded_coords[0], expanded_coords[1]),
                                (expanded_coords[2], expanded_coords[3]),
                                (0, 255, 255), 1)  # Yellow for possession area
                
                # Draw team number and ball possession indicator
                label = f"Team {team}"
                if has_ball:
                    label += " (Ball)"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                               
        # Draw possession statistics
        self.draw_possession_stats(frame)
        
        return frame
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better color detection
        """
        # Convert to LAB color space for better color enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def expand_bbox(self, frame_shape, bbox, scale=1.5):
        """
        Expand bounding box while keeping it within frame boundaries
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Calculate new coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        new_x1 = max(0, center_x - new_width // 2)
        new_y1 = max(0, center_y - new_height // 2)
        new_x2 = min(frame_shape[1], center_x + new_width // 2)
        new_y2 = min(frame_shape[0], center_y + new_height // 2)
        
        return [new_x1, new_y1, new_x2, new_y2]

    def draw_possession_stats(self, frame):
        """
        Draw possession statistics on frame
        """
        # Calculate possession percentages
        total = self.total_frames if self.total_frames > 0 else 1
        team1_possession = (self.frames_equipo_1 / total) * 100
        team2_possession = (self.frames_equipo_2 / total) * 100
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        
        # Draw title
        cv2.putText(frame, "Possession Statistics", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        # Draw team 1 stats
        cv2.putText(frame, f"Team 1: {team1_possession:.1f}%", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Draw team 2 stats
        cv2.putText(frame, f"Team 2: {team2_possession:.1f}%", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
    

    def guardar_frame(self, frame, frame_index=0):
        """
        Save frame to disk
        """
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