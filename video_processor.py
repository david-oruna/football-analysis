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

        # Add team color tracking
        self.team1_color = None
        self.team2_color = None
        self.color_confidence = 0
        self.min_players_for_team = 3  # Minimum players needed to establish team colors

        
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
        original_height, original_width = frame.shape[:2]
        frame = cv2.resize(frame, (int(original_width * self.scale_factor), int(original_height * self.scale_factor)))

        # Process person detections
        results = self.yolo_model(frame, classes=[0], conf=0.25)
        
        # Process ball detections
        ball_results = self.model_pelota(frame, classes=[32], conf=0.1)
        frame_with_detections = frame.copy()
        
        # Get ball bbox if detected
        ball_bbox = np.array([0, 0, 0, 0])
        if len(ball_results[0].boxes) > 0:
            ball_box = ball_results[0].boxes[0]
            if len(ball_box.xyxy) > 0:
                ball_bbox = ball_box.xyxy[0].cpu().numpy()

        player_detections = []
        
        # Process all detected players
        for result in results:
            detections = result.boxes
            processed_frame = self.preprocess_frame(frame)
            
            for detection in detections:
                if len(detection.xyxy) >= 1:
                    bbox = detection.xyxy[0].cpu().numpy().astype(int)
                    player_info = self.process_player(processed_frame, ball_bbox, bbox)
                    if player_info:
                        player_detections.append(player_info)

        # Only process team assignments if we have players
        if player_detections:
            self.assign_teams(player_detections)
            # Update possession stats before drawing
            self.update_possession(player_detections)
            frame_with_detections = self.draw_results(frame_with_detections, player_detections)

        # Draw possession stats
        frame_with_detections = self.draw_possession_stats(frame_with_detections)
        
        cv2.imshow('Detecciones', frame_with_detections)
        return frame_with_detections
        


    def process_player(self, frame, ball_bbox, player_bbox):
        """
        Modified to return ball position when possession is detected
        """
        try:
            x1, y1, x2, y2 = player_bbox
            
            # Basic validation
            if x2 <= x1 or y2 <= y1:
                if self.depurar:
                    print("Invalid bbox dimensions")
                return None
                
            # Extract jersey region (upper body)
            body_height = y2 - y1
            jersey_y1 = int(y1 + body_height * 0.15)  # Start slightly higher
            jersey_y2 = int(y1 + body_height * 0.45)  # End before middle
            jersey_x1 = int(x1 + (x2 - x1) * 0.1)  # Trim sides to avoid arms
            jersey_x2 = int(x2 - (x2 - x1) * 0.1)
            
            # Ensure coordinates are within frame bounds
            jersey_y1 = max(0, jersey_y1)
            jersey_y2 = min(frame.shape[0], jersey_y2)
            jersey_x1 = max(0, jersey_x1)
            jersey_x2 = min(frame.shape[1], jersey_x2)
            
            if jersey_y2 <= jersey_y1 or jersey_x2 <= jersey_x1:
                if self.depurar:
                    print("Invalid jersey region")
                return None
            
            jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
            
            if jersey_region.size == 0:
                if self.depurar:
                    print("Empty jersey region")
                return None
                
            # Get dominant colors
            jersey_color = self.get_dominant_color(jersey_region)
            if jersey_color is None:
                if self.depurar:
                    print("No dominant color found")
                return None

            # Calculate expanded region for ball detection
            mid_coords = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
            half_dims = np.array([(x2 - x1) // 2, (y2 - y1) // 2])
            expanded_half_dims = (half_dims * 3).astype(int)
            
            expanded_coords = np.array([
                max(0, mid_coords[0] - expanded_half_dims[0]),
                max(0, mid_coords[1] - expanded_half_dims[1]),
                min(frame.shape[1], mid_coords[0] + expanded_half_dims[0]),
                min(frame.shape[0], mid_coords[1] + expanded_half_dims[1])
            ])
            
            has_ball, ball_pos = self.ball_possession(ball_bbox, expanded_coords)
            
            if self.depurar:
                # Draw jersey region for debugging
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (jersey_x1, jersey_y1), (jersey_x2, jersey_y2), (0, 255, 0), 2)
                cv2.imshow('Jersey Region', debug_frame)
            
            return {
                'bbox': player_bbox,
                'jersey_color': jersey_color,
                'has_ball': has_ball,
                'ball_position': ball_pos,
                'player_center': (int((x1 + x2) // 2), int((y1 + y2) // 2))  # Add player center
            }
            
        except Exception as e:
            if self.depurar:
                print(f"Error in process_player: {str(e)}")
            return None

    def ball_possession(self, ball_bbox, expanded_coords):
        """
        Enhanced ball detection using distance calculation
        """
        if ball_bbox.any():
            ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
            ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
            
            # Calculate relative position in expanded region
            if (expanded_coords[0] <= ball_center_x <= expanded_coords[2] and
                expanded_coords[1] <= ball_center_y <= expanded_coords[3]):
                
                # Store ball position for drawing
                return True, (int(ball_center_x), int(ball_center_y))
        
        return False, None

    def get_dominant_color(self, image):
        """
        More robust color detection using k-means
        """
        try:
            if image.size == 0 or image is None:
                return None

            # Resize image for faster processing
            height, width = image.shape[:2]
            if height * width > 10000:  # If image is too large
                scale = np.sqrt(10000 / (height * width))
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size)

            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Create mask for valid colors (exclude very dark/light colors)
            lower_bound = np.array([0, 30, 30])
            upper_bound = np.array([180, 255, 255])
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

            # Get valid pixels
            valid_pixels = hsv_image[mask > 0]

            if len(valid_pixels) < 100:  # Need minimum number of valid pixels
                return None

            # Reshape for k-means
            pixels = valid_pixels.reshape(-1, 3)

            # Apply k-means clustering
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0]

            # Convert back to BGR for visualization
            dominant_color_bgr = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_HSV2BGR)[0][0]

            if self.depurar:
                print(f"Dominant color (HSV): {dominant_color}")
                print(f"Dominant color (BGR): {dominant_color_bgr}")

            return dominant_color

        except Exception as e:
            if self.depurar:
                print(f"Error in get_dominant_color: {str(e)}")
            return None

    def assign_teams(self, player_detections):
        """
        Improved team assignment with better clustering
        """
        jersey_colors = [p['jersey_color'] for p in player_detections if p['jersey_color'] is not None]
        
        if len(jersey_colors) < self.min_players_for_team:
            if self.team1_color is not None and self.team2_color is not None:
                self._assign_players_to_teams(player_detections)
            return
        
        colors = np.array(jersey_colors)
        
        # Perform clustering if needed
        if self.team1_color is None or self.team2_color is None or self.color_confidence < 5:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(colors)
            
            cluster_labels = kmeans.labels_
            cluster_sizes = np.bincount(cluster_labels)
            
            if min(cluster_sizes) >= self.min_players_for_team:
                new_centers = kmeans.cluster_centers_
                
                if self.team1_color is None:
                    self.team1_color = new_centers[0]
                    self.team2_color = new_centers[1]
                else:
                    # Smooth transition with better color matching
                    alpha = 0.8
                    dist1 = np.linalg.norm(new_centers[0] - self.team1_color)
                    dist2 = np.linalg.norm(new_centers[0] - self.team2_color)
                    
                    if dist1 < dist2:
                        self.team1_color = alpha * self.team1_color + (1 - alpha) * new_centers[0]
                        self.team2_color = alpha * self.team2_color + (1 - alpha) * new_centers[1]
                    else:
                        self.team1_color = alpha * self.team1_color + (1 - alpha) * new_centers[1]
                        self.team2_color = alpha * self.team2_color + (1 - alpha) * new_centers[0]
                
                self.color_confidence += 1
        
        self._assign_players_to_teams(player_detections)

    def _assign_players_to_teams(self, player_detections):
        """
        Assign players to teams using established team colors
        """
        for player in player_detections:
            if player['jersey_color'] is not None:
                dist1 = np.linalg.norm(player['jersey_color'] - self.team1_color)
                dist2 = np.linalg.norm(player['jersey_color'] - self.team2_color)
                player['team'] = 1 if dist1 < dist2 else 2

    def update_possession(self, player_detections):
        """
        Updated possession calculation
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
        elif team2_has_ball:
            self.frames_equipo_2 += 1
        
        # Always increment total frames if we have any detections
        if player_detections:
            self.total_frames += 1

    
    def draw_results(self, frame, player_detections):
        """
        Modified drawing function to draw lines instead of boxes for ball possession
        """
        for player in player_detections:
            bbox = player['bbox']
            team = player.get('team')
            has_ball = player.get('has_ball', False)
            
            # Set team colors
            if team == 1:
                color = (0, 0, 255)  # Red for team 1
            elif team == 2:
                color = (255, 0, 0)  # Blue for team 2
            else:
                color = (0, 255, 0)  # Green for unassigned
                
            # Draw player box
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw team label
            label = f"Team {team}" if team else "Unknown"
            
            # Draw ball possession line
            if has_ball and player.get('ball_position') is not None:
                ball_pos = player['ball_position']
                player_center = player['player_center']
                
                # Draw line from ball to player
                cv2.line(frame, 
                        ball_pos, 
                        player_center,
                        (0, 255, 255),  # Yellow line
                        2)  # Line thickness
                
                # Add arrow head
                self.draw_arrow_head(frame, 
                                   ball_pos, 
                                   player_center,
                                   (0, 255, 255),
                                   15)  # Arrow size
                
                label += " (Ball)"
            
            # Draw label
            cv2.putText(frame, label, 
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
        
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
        Fixed possession stats drawing
        """
        if self.total_frames == 0:
            return frame
            
        # Calculate possession percentages
        team1_possession = (self.frames_equipo_1 / max(self.total_frames, 1)) * 100
        team2_possession = (self.frames_equipo_2 / max(self.total_frames, 1)) * 100
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        cv2.putText(frame, "Possession Statistics", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Team 1 (Red): {team1_possession:.1f}%", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Team 2 (Blue): {team2_possession:.1f}%", 
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)

        # Debug print
        if self.depurar:
            print(f"Frames - Team1: {self.frames_equipo_1}, Team2: {self.frames_equipo_2}, Total: {self.total_frames}")
            print(f"Possession - Team1: {team1_possession:.1f}%, Team2: {team2_possession:.1f}%")
        
        return frame

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

    def draw_arrow_head(self, frame, start_point, end_point, color, size):
        """
        Helper function to draw arrow head
        """
        # Calculate arrow direction
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = np.arctan2(dy, dx)
        
        # Calculate arrow head points
        angle_left = angle + np.pi/6
        angle_right = angle - np.pi/6
        
        pt_left = (
            int(end_point[0] - size * np.cos(angle_left)),
            int(end_point[1] - size * np.sin(angle_left))
        )
        
        pt_right = (
            int(end_point[0] - size * np.cos(angle_right)),
            int(end_point[1] - size * np.sin(angle_right))
        )
        
        # Draw arrow head lines
        cv2.line(frame, end_point, pt_left, color, 2)
        cv2.line(frame, end_point, pt_right, color, 2)