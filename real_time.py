# -*- coding: utf-8 -*-
import numpy as np
import cv2
from os import listdir, getcwd, remove
from os.path import join
from time import sleep

#reproducimos el video y modificamos características con OpenCV
def play_video():
    cwd = getcwd()
    vid_path = join(cwd,'video_1.mp4')
    cap = cv2.VideoCapture(vid_path)
    
    if (cap.isOpened()== False): 
        print("Error al abrir el vídeo")
    
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cropped = frame[266:582]
        resized = cv2.resize(cropped, (cropped.shape[1]+160,cropped.shape[0]+160), interpolation = cv2.INTER_AREA)
        cropped_domino = find_dominoes(resized)
        img_with_blobs, centers = get_shot_blob(cropped_domino,
                                            filter_by_area = True,
                                            A = (1,500),
                                            filter_by_color = False,
                                            filter_by_circularity = True,
                                            min_circularity = 0.85,
                                            filter_by_covexity = True,
                                            min_convexity = 0.9, 
                                            min_dist_between_blobs = 0,
                                            filter_by_inertia = False,
                                            min_inertia_ratio = 0.15 
                                            )
        img_with_slope, slope_prep, prep_point = get_domino_slope(img_with_blobs)
        if slope_prep == 100:
            img_with_txt = add_text(resized, str(0))
        else:
            upper, lower = count_dots(slope_prep, prep_point, centers)
            img_with_blobs, _ = get_shot_blob(resized,
                                            filter_by_area = True,
                                            A = (1,500),
                                            filter_by_color = False,
                                            filter_by_circularity = True,
                                            min_circularity = 0.85,
                                            filter_by_covexity = True,
                                            min_convexity = 0.9, 
                                            min_dist_between_blobs = 0,
                                            filter_by_inertia = False,
                                            min_inertia_ratio = 0.15 
                                            )
            img_with_txt = add_text(img_with_blobs, f'({str(upper)}, {str(lower)})')
        
        cv2.imshow('video',img_with_txt)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        i+=1

    cap.release()
    cv2.destroyAllWindows()

#definimos el contador
def add_text(img, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,40)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    img_with_txt = cv2.putText(img,text, 
                               bottomLeftCornerOfText, 
                               font, 
                               fontScale,
                               fontColor,
                               lineType)
    return img_with_txt

#definimos la busqueda de las piezas
def find_dominoes(image):
    
    image_copy = image.copy()

    rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)

    light_white = (0, 0, 40)
    dark_white = (200,200,255)
        
    mask_white = cv2.inRange(hsv, light_white, dark_white)
    result_white = cv2.bitwise_and(image_copy, image_copy, mask = mask_white)

    kernel = np.ones((3, 3), np.uint8)
    opened_1 = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel,iterations = 2)
    opened_1 = cv2.dilate(opened_1, kernel, iterations = 15)
    opened_1 = cv2.morphologyEx(opened_1, cv2.MORPH_OPEN, kernel,iterations = 1)
        
    opened_2 = cv2.erode(opened_1, kernel, iterations = 3)
    opened_3 = cv2.dilate(opened_2, kernel, iterations = 5)
    mask = cv2.erode(opened_3, kernel, iterations = 20)

    contours, _ = cv2.findContours(opened_3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Creamos la máscara donde el blanco es lo que queremos
    mask = np.zeros_like(image_copy)
#Rellenamos el contorno de la máscara
    cv2.drawContours(mask, contours, 0, 255, -1) 
#Extraemos el objeto y lo mostramos por pantalla
    out = np.zeros_like(image_copy)
    out[mask == 255] = image_copy[mask == 255]
    
#Recortamos
    (y, x) = np.where(mask == 255)[0:2]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = image_copy[topy:bottomy+1, topx:bottomx+1]
    return out

#definimos una función para contar los puntos en la ficha. 
#Para ello utilizaremos el método de detección de BLOB y ajustaremos los parámetros mediante prueba y error.
def get_shot_blob(image,
                  filter_by_area = True,
                  A = (1,150),
                  filter_by_color = True,
                  filter_by_circularity = True,
                  min_circularity = 0.8, # 0.8
                  filter_by_covexity = True,
                  min_convexity = 0.9, # 0.94
                  min_dist_between_blobs = 0,
                  filter_by_inertia = True,
                  min_inertia_ratio = 0.15 # 0.66 
                 ):
#toma una imagen RGB, devuelve los centros de las tomas mediante un simple método de detección de manchas
    image_copy = image.copy()
    img_Gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    
# Cambiar umbrales
    params.minThreshold = 0
    params.maxThreshold = 500
    
# Filtramos por áreas
    if filter_by_area:
        params.filterByArea = filter_by_area    
        params.minArea = A[0]
        params.maxArea = A[1]
# Filtramos por color.
    if filter_by_color:
        params.filterByColor = filter_by_color
        params.blobColor = 0

# Filtrar por forma circular
    if filter_by_circularity:
        params.filterByCircularity = filter_by_circularity
        params.minCircularity = min_circularity 

# Filtrar por convexidad
    if filter_by_covexity:
        params.filterByConvexity = filter_by_covexity
        params.minConvexity = min_convexity

    params.minDistBetweenBlobs = min_dist_between_blobs

# Filtrar por Inercia
    if filter_by_inertia:
        params.filterByInertia = filter_by_inertia
        params.minInertiaRatio = min_inertia_ratio
    
# Creamos un detector con los parámetros
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    im_with_keypoints = detector.detect(img_Gray)
    
    centers = []
    for key in im_with_keypoints:
        centers.append((int(key.pt[0]), int(key.pt[1])))
    
# Resaltar puntos.
    for center in centers:
        output_image = cv2.circle(image_copy, center, 10,(0,255,0), -1)

    if len(centers) != 0:
        return output_image, centers
    else:
        return image, centers
#definir una función para encontrar la pendiente de la línea
def get_domino_slope(image):
    
    image_copy = image.copy()
    dst = cv2.Canny(image_copy, 0, 200, None, 3)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 100)

    if linesP is None:
        return image, 100, (0,0)

# Calcular longitudes de líneas detectadas
    lengthes = []
    for x1,y1,x2,y2 in linesP[:, 0]:
        lengthes.append(((x1-x2)**2+(y1-y2)**2)**0.5)
    
# Extrae las coordenadas de la línea más larga
    x1,y1,x2,y2 = linesP[lengthes == max(lengthes),0][0]
    
# Dibujar la línea más larga
    image_copy = image.copy()
    out = cv2.line(image_copy,(x1,y1),(x2,y2),(0,255,0),2)
    
# Encuentra el centro de la línea más larga.
    x_center = int((x1+x2)/2)
    y_center = int((y1+y2)/2)
    prep_point = (x_center, y_center)
# Calcular la pendiente de la recta prependicular a la recta más larga
    if (y1-y2) == 0 or (x1-x2) == 0:
        return image, 100, (0,0)
    slope = (y1-y2)/(x1-x2)
    slope_prep = -1/slope
    
# Calcular otro punto en la recta perpendicular
    x_prep = 100
    y_prep = int(y_center + (slope_prep*(x_prep - x_center)))
    
# Dibujar la linea perpendicular
    out = cv2.line(image_copy, (x_prep,y_prep), (x_center,y_center), (0,255,0), 2)
    out = cv2.circle(image_copy, (x_center,y_center), 10,(0,0,255), -1)
    out = cv2.circle(image_copy, (x_prep,y_prep), 10,(0,0,255), -1)
        
    return out, slope_prep, prep_point

#definimos una funcion para contar los puntos a ambos lados de la linea perpendicular
def count_dots(slope, point, centers):
    upper = 0
    lower = 0
    x_point = point[0]
    y_point = point[1]

    for center in centers:
        x = center[0]
        y = center[1]
        
        y_line = y_point + (slope*(x - x_point))
        if y > y_line:
            lower += 1
        else:
            upper += 1
    print(f'Número de puntos: ({upper}, {lower})')
    return upper, lower

if __name__ == '__main__':
    play_video()