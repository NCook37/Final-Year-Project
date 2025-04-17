import cv2
import numpy as np
import dlib
import skimage
from matplotlib import pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from matplotlib import cm

def filter_image(image, display_steps):
    eye_image = image #skimage.io.imread("C:/Users/nicho/OneDrive/Pictures/Eye pictures/eye50.jpg")

    roi_eye = cv2.GaussianBlur(eye_image, (3,3), 0)

    grey = np.invert(cv2.cvtColor(roi_eye, cv2.COLOR_BGR2GRAY))

    mid = np.round(grey.max() * 0.95)

    _, threshold = cv2.threshold(grey, mid, 255, cv2.THRESH_BINARY)

    if(display_steps):
        plt.imshow(eye_image,cmap="gray")
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

        plt.imshow(threshold,cmap="gray")
        plt.title("Threshold Image")
        plt.axis('off')
        plt.show()
    
    return threshold


def calculate_average(possible_x_coordinates, possible_y_coordinates):

    mean_x = sum(possible_x_coordinates) / len(possible_x_coordinates)
    deviation_x = (np.std(possible_x_coordinates))

    refined_x = 0
    outliers_x = 0

    for i in range(0,len(possible_x_coordinates)):
        value = possible_x_coordinates[i]
        if(value > mean_x - 2*deviation_x and value < mean_x + 2*deviation_x):
            refined_x += value
        else:
            outliers_x += 1

    mean_y = sum(possible_y_coordinates) / len(possible_y_coordinates)
    deviation_y = (np.std(possible_y_coordinates))

    refined_y = 0
    outliers_y = 0

    for i in range(0,len(possible_y_coordinates)):
        value = possible_y_coordinates[i]
        if(value > mean_y - 2*deviation_y and value < mean_y + 2*deviation_y):
            refined_y += value
        else:
            outliers_y += 1

    predicted_x = np.round(refined_x / (len(possible_x_coordinates)-outliers_x))
    predicted_y = np.round(refined_y / (len(possible_y_coordinates)-outliers_y))
    return(predicted_x, predicted_y)

def find_intersection_point(line1,line2):
    theta1, rho1 = line1[0], line1[1]
    theta2, rho2 = line2[0], line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ]) 
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        # Round up x and y because pixel cannot have float number
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        if x0 < 0 or y0 < 0:
            return None
        return x0, y0
    else:
        return None

def find_pupil(image, display_steps):
    #fig, axes = plt.subplots(1, 3,figsize=(16, 5))
    #ax = axes.ravel()
    
    changed_image = image
    #if(display_steps):
        #ax[0].imshow(image, cmap=cm.gray)
        #ax[0].set_title('Detected Edges')
        #ax[0].set_axis_off()

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(changed_image, theta=tested_angles)
    lines = hough_line_peaks(h,theta,d,threshold=1,num_peaks=30)

    angles, dists = lines[1], lines[2]

    #for i in range(0,len(angles)):
        #(x0, y0) = dists[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
        #ax[1].axline((x0, y0),slope=np.tan(angles[i] + np.pi/2), color="red", linewidth=1)

    #ax[1].set_ylim((changed_image.shape[0], 0))
    #ax[1].set_axis_off()
    #ax[1].set_title('Detected lines')

    if(display_steps):
        ax[1].imshow(changed_image, cmap=cm.gray)

    possible_x_coordinates = []
    possible_y_coordinates = []

    for i in range(1, (len(angles) - 1)):
        line1 = angles[i -1], dists[i-1]
        line2 = angles[i], dists[i]
        coordinate = find_intersection_point(line1, line2)
        if coordinate is not None:
            possible_x, possible_y = coordinate
            #ax[2].plot(possible_x, possible_y, 'ro')
            possible_x_coordinates.append(possible_x)
            possible_y_coordinates.append(possible_y)

    #if(display_steps):
        #ax[2].imshow(changed_image, cmap=cm.gray)
        #plt.show()

    if(not possible_x_coordinates or not possible_y_coordinates):
        predicted_x, predicted_y = 0,0
    else:    
        predicted_x, predicted_y = calculate_average(possible_x_coordinates, possible_y_coordinates)
    return(predicted_x, predicted_y)

class Eye(object):
    def __init__(self):
        self.pupil = (None, None)
        self.centre = (None, None)
        self.pupilFound = False
        self.height = 0
        self.width = 0

def left_eye(frame, landmarks):
    x_min = min(landmarks.part(36).x,landmarks.part(37).x, landmarks.part(41).x)
    y_min = min(landmarks.part(36).y,landmarks.part(37).y, landmarks.part(38).y)
    x_max = max(landmarks.part(38).x,landmarks.part(39).x, landmarks.part(40).x)
    y_max = max(landmarks.part(39).y,landmarks.part(40).y, landmarks.part(41).y)

    eye = frame[y_min:y_max, x_min:x_max]

    global prev_left_x
    global prev_left_y

    eye_grey = filter_image(eye, False)
    eye = Eye()
    try:
        possible_x, possible_y = find_pupil(eye_grey, False)
        prev_left_x, prev_left_y = possible_x, possible_y
        frame[(y_min + int(possible_y)), (x_min + int(possible_x))] = [0,0,255]
        eye.pupil = (x_min + possible_x,y_min + possible_y)
        eye.centre = ((x_max + x_min) / 2, (y_max + y_min) / 2)
        eye.height = y_max - y_min
        eye.width = x_max - x_min
        eye.pupilFound = True
    except:
        print("Failed to detect pupil left")
    return eye

def right_eye(frame, landmarks):
    x_min = min(landmarks.part(42).x,landmarks.part(43).x, landmarks.part(47).x)
    y_min = min(landmarks.part(42).y,landmarks.part(43).y, landmarks.part(44).y)
    x_max = max(landmarks.part(44).x,landmarks.part(45).x, landmarks.part(46).x)
    y_max = max(landmarks.part(42).y,landmarks.part(47).y, landmarks.part(46).y)

    eye = frame[y_min:y_max, x_min:x_max]

    global prev_right_x
    global prev_right_y
    eye_grey = filter_image(eye, False)
    eye = Eye()
    try:
        possible_x, possible_y = find_pupil(eye_grey, False)
        prev_right_x, prev_right_y = possible_x, possible_y
        frame[(y_min + int(possible_y)), (x_min + int(possible_x))] = [0,0,255]
        eye.pupil = (x_min + possible_x,y_min + possible_y)
        eye.centre = ((x_max + x_min) / 2, (y_max + y_min) / 2)
        eye.height = y_max - y_min
        eye.width = x_max - x_min
        eye.pupilFound = True
    except:
        print("Failed to detect pupil right")
    return eye

def scale_down_values(x, y, width, height):
    return((x/width, y/height))

def scale_up_values(x, y, width, height):
    return((int(x*width), int(y*height)))