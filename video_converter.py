import os
import cv2

def crear_video_desde_imagenes(ruta_carpeta, nombre_video_salida, fps=30):
    # Obtener la lista de archivos de imagen en la carpeta
    imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.png')],
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Ordenar por el índice del frame

    # Verificar que haya imágenes
    if not imagenes:
        print("No se encontraron imágenes en la carpeta especificada.")
        return

    # Leer la primera imagen para obtener las dimensiones
    primera_imagen = cv2.imread(os.path.join(ruta_carpeta, imagenes[0]))
    if primera_imagen is None:
        print("Error al leer la primera imagen.")
        return

    altura, ancho, _ = primera_imagen.shape

    # Definir el codec y crear el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'H264') 
    video_salida = cv2.VideoWriter(nombre_video_salida, fourcc, fps, (ancho, altura))

    # Agregar cada imagen al video
    for imagen in imagenes:
        ruta_imagen = os.path.join(ruta_carpeta, imagen)
        frame = cv2.imread(ruta_imagen)
        if frame is not None:
            video_salida.write(frame)  # Agregar el frame al video
        else:
            print(f"Error al leer la imagen: {ruta_imagen}")

    # Liberar el objeto VideoWriter
    video_salida.release()
    print(f"Video creado correctamente: {nombre_video_salida}")