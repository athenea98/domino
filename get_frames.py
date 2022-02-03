import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir, getcwd, remove
from os.path import join
from time import sleep

def get_frames():
    cwd = getcwd()
    vid_path = join(cwd,'video.mp4')
    cap = cv2.VideoCapture(vid_path)

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(join(cwd,'frames', 'f'+str(i)+'.jpg'), frame)
        i+=1

    cap.release()
    cv2.destroyAllWindows()

    for j in range(675):
        if j < 10:
            remove(join(cwd,'frames', 'f'+str(j)+'.jpg'))

        elif (j-10)%75 == 0:
            img = cv2.imread(join(cwd,'frames', 'f'+str(j)+'.jpg'))
            cropped = img[266:582]
            resized = cv2.resize(cropped, (cropped.shape[1]+160,cropped.shape[0]+160), interpolation = cv2.INTER_AREA)
            cv2.imwrite(join(cwd,'frames', 'f'+str(j)+'.jpg'), resized)
            
        else:
            remove(join(cwd,'frames', 'f'+str(j)+'.jpg'))
    
    images = [f for f in listdir(join(cwd, 'frames'))]
    images.insert(1,images[-1])
    images = images[:-1]

    return images

cwd = getcwd()
images = get_frames()
print(images)
img = cv2.imread(join(cwd, 'frames', images[7]))
plt.imshow(img)
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
    
    mask = np.zeros_like(image_copy) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(image_copy) # Extract out the object and place into output image
    out[mask == 255] = image_copy[mask == 255]
    
    # Now crop
    (y, x) = np.where(mask == 255)[0:2]
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = image_copy[topy:bottomy+1, topx:bottomx+1]
    return out
cropped_img = find_dominoes(img)
plt.imshow(cropped_img)

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
    ## takes RGB image , returns centers of shots  by simple blob detector method
    image_copy = image.copy()
    img_Gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 500

    # Filter by Area.
    if filter_by_area:
        params.filterByArea = filter_by_area    
        params.minArea = A[0]
        params.maxArea = A[1]
    # Filter by Color.
    if filter_by_color:
        params.filterByColor = filter_by_color
        params.blobColor = 0

    # Filter by Circularity
    if filter_by_circularity:
        params.filterByCircularity = filter_by_circularity
        params.minCircularity = min_circularity 

    # Filter by Convexity
    if filter_by_covexity:
        params.filterByConvexity = filter_by_covexity
        params.minConvexity = min_convexity

    params.minDistBetweenBlobs = min_dist_between_blobs

    # Filter by Inertia
    if filter_by_inertia:
        params.filterByInertia = filter_by_inertia
        params.minInertiaRatio = min_inertia_ratio
    
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    im_with_keypoints = detector.detect(img_Gray)
#     print(len(im_with_keypoints))
    
    centers = []
    for key in im_with_keypoints:
        centers.append((int(key.pt[0]), int(key.pt[1])))
    
    # Highlight dots.
    for center in centers:
        output_image = cv2.circle(image_copy, center, 10,(0,255,0), -1)
        
    return output_image, centers
output_image, centers = get_shot_blob(cropped_img,
                          filter_by_area = True,
                          A = (1,500),
                          filter_by_color = False,
                          filter_by_circularity = True,
                          min_circularity = 0.85,
                          filter_by_covexity = True,
                          min_convexity = 0.9, 
                          min_dist_between_blobs = 0,
                          filter_by_inertia = False,
                          min_inertia_ratio = 0.15 # 0.66 
                         )
plt.imshow(output_image)

def get_domino_slope(image):
    
    image_copy = image.copy()
    dst = cv2.Canny(image_copy, 0, 200, None, 3)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 100)

    # Calculate lengthes of detected lines
    lengthes = []
    for x1,y1,x2,y2 in linesP[:, 0]:
        lengthes.append(((x1-x2)**2+(y1-y2)**2)**0.5)
    
    # Extract coordinates of the longest line
    x1,y1,x2,y2 = linesP[lengthes == max(lengthes),0][0]
    
    # Draw the longest line
    image_copy = image.copy()
    out = cv2.line(image_copy,(x1,y1),(x2,y2),(0,255,0),2)
    
    # Find the center of the longest line
    x_center = int((x1+x2)/2)
    y_center = int((y1+y2)/2)
    prep_point = (x_center, y_center)
    
    # Calculate the slope of the prependicular to the longest line
    slope = (y1-y2)/(x1-x2)
    slope_prep = -1/slope
    
    # Calculate another point on the prependicular line
    x_prep = 100
    y_prep = int(y_center + (slope_prep*(x_prep - x_center)))
    
    # Draw the prependicular line
    out = cv2.line(image_copy, (x_prep,y_prep), (x_center,y_center), (0,255,0), 2)
    out = cv2.circle(image_copy, (x_center,y_center), 10,(0,0,255), -1)
    out = cv2.circle(image_copy, (x_prep,y_prep), 10,(0,0,255), -1)
        
    return out, slope_prep, prep_point
lined_image, slope_prep, prep_point = get_domino_slope(cropped_img)
plt.imshow(lined_image)
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
    print(f'Number of dots: ({upper}, {lower})')
    
count_dots(slope_prep, prep_point, centers)
num_images = len(images)

fig, axes = plt.subplots(3, num_images, figsize = [20, 15])

for i,img_name in enumerate(images):
    img = cv2.imread(join(cwd, 'frames', img_name))
    cropped_img = find_dominoes(img)
    plt.subplot(3, num_images, i+1)
    plt.imshow(cropped_img)
    
    dotted_img, centers = get_shot_blob(cropped_img,
                          filter_by_area = True,
                          A = (1,500),
                          filter_by_color = False,
                          filter_by_circularity = True,
                          min_circularity = 0.85,
                          filter_by_covexity = True,
                          min_convexity = 0.9, 
                          min_dist_between_blobs = 0,
                          filter_by_inertia = False,
                          min_inertia_ratio = 0.15 # 0.66 
                         )
    plt.subplot(3, num_images, i+1+num_images)
    plt.imshow(dotted_img)
    
    lined_img, slope_prep, prep_point = get_domino_slope(cropped_img)
    plt.subplot(3, num_images, i+1+2*num_images)
    plt.imshow(lined_img)
    
    count_dots(slope_prep, prep_point, centers)

