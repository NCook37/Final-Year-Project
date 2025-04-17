import numpy as np
import cv2
import dlib
import torch
import torch.optim as optim
import torch.nn as nn
import pickle
from eye_tracking import *
import screeninfo
from pathlib import Path

path = str(Path.cwd())

face_cascade = cv2.CascadeClassifier(path + "/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(path + "/haarcascade_eye.xml")

class Seven_layer_network(nn.Module):
    def __init__(self, rng, D_in, D_hid_1, D_hid_2, D_hid_3, D_hid_4, D_hid_5):
        super(Seven_layer_network, self).__init__()
        # Construct and initialize network parameters
        D_in = D_in # Dimension of input features
        D_out = 2 # Dimension of Output layer.
        self.layer_1 = nn.Linear(D_in+1, D_hid_1)
        self.bn1 = nn.BatchNorm1d(D_hid_1)
        self.layer_2 = nn.Linear(D_hid_1+1, D_hid_2)
        self.bn2 = nn.BatchNorm1d(D_hid_2)
        self.layer_3 = nn.Linear(D_hid_2+1, D_hid_3)
        self.bn3 = nn.BatchNorm1d(D_hid_3)
        self.layer_4 = nn.Linear(D_hid_3+1, D_hid_4)
        self.bn4 = nn.BatchNorm1d(D_hid_4)
        self.layer_5 = nn.Linear(D_hid_4+1, D_hid_5)
        self.bn5 = nn.BatchNorm1d(D_hid_5)
        self.layer_6 = nn.Linear(D_hid_5+1, D_out)
        
        #self.loss_function = MSE_loss()
        ###########################################################################
    
    def backward_pass(self, loss):
        # Performs back propagation and computes gradients
        # With PyTorch, we do not need to compute gradients analytically for parameters were requires_grads=True, 
        # Calling loss.backward(), torch's Autograd automatically computes grads of loss wrt each parameter p,...
        # ... and **puts them in p.grad**. Return them in a list.
        loss.backward()
        grads = [param.grad for param in self.parameters()]
        return grads
        
    def forward_pass(self, batch_inputs):
        # Get parameters
        
        batch_inputs_t = torch.tensor(batch_inputs, dtype=torch.float)  # Makes pytorch array to pytorch tensor.
        
        unary_feature_for_bias = torch.ones(size=(batch_inputs.shape[0], 1)) # [N, 1] column vector.
        x = torch.cat((batch_inputs_t, unary_feature_for_bias), dim=1) # Extra feature=1 for bias.
    
        # Layer 1
        h1_preact = self.layer_1(x) #.mm(w1)
        h1_act = h1_preact.clamp(min=0)
        h1_act = self.bn1(h1_act)
        # Layer 2:
        h1_ext = torch.cat((h1_act, unary_feature_for_bias), dim=1)
        h2_preact = self.layer_2(h1_ext)
        h2_act = h2_preact.clamp(min=0)
        h2_act = self.bn2(h2_act)
        # Layer 3:
        h2_ext = torch.cat((h2_act, unary_feature_for_bias), dim=1)
        h3_preact = self.layer_3(h2_ext)
        h3_act = h3_preact.clamp(min=0)
        h3_act = self.bn3(h3_act)
        # Layer 4:
        h3_ext = torch.cat((h3_act, unary_feature_for_bias), dim=1)
        h4_preact = self.layer_4(h3_ext)
        h4_act = h4_preact.clamp(min=0)
        h4_act = self.bn4(h4_act)

        h4_ext = torch.cat((h4_act, unary_feature_for_bias), dim=1)
        h5_preact = self.layer_5(h4_ext)
        h5_act = h5_preact.clamp(min=0)
        h5_act = self.bn5(h5_act)

        h5_ext = torch.cat((h5_act, unary_feature_for_bias), dim=1)
        h6_preact = self.layer_6(h5_ext)
        h6_act = h6_preact
        x_pred = h6_act

                
        return x_pred
        
        
def Magnitude_loss(x_pred, x_real, eps=1e-7):
    # Cross entropy: See Lecture 5, slide 19.
    # x_pred: [N, D_out] Prediction returned by forward_pass. Numpy array of shape [N, D_out]
    # x_real: [N, D_in]
    
    # If number array is given, change it to a Torch tensor.
    x_pred = torch.tensor(x_pred, dtype=torch.float) if type(x_pred) is np.ndarray else x_pred
    x_real = torch.tensor(x_real, dtype=torch.float) if type(x_real) is np.ndarray else x_real

    loss_recon = torch.sqrt(torch.sum(torch.square((x_pred - x_real)), axis= 1))
    cost = torch.mean(loss_recon) # Expectation of loss: Mean over samples (axis=0).
    return cost

def MSE_loss(x_pred, x_real):
    mse = torch.mean(torch.square(x_pred - torch.tensor(x_real)))
    with torch.no_grad():
        return mse.numpy()

def MAE_loss(x_pred, x_real):
    mae = torch.mean(abs(x_pred - torch.tensor(x_real)))
    with torch.no_grad():
        return mae.numpy()

trained_model = pickle.load(open(path + '/trained model.sav', 'rb'))
trained_model.eval()

monitor = screeninfo.get_monitors()[0]
screen_width_mm = monitor.width_mm
screen_height_mm = monitor.height_mm
screen_width = monitor.width
screen_height = monitor.height

trajectory_points = []
traj_x_min = 0.05
traj_x_max = 0.95
x_values = np.linspace(traj_x_min, traj_x_max, num=360)

traj_y_min = 0.1
traj_y_max = 0.3
y_values = np.linspace(traj_y_min, traj_y_max, num=80)
for value in x_values:
    trajectory_points.append([value, traj_y_min])
for value in y_values:
    trajectory_points.append([traj_x_max, value])
for value in x_values:
    trajectory_points.append([1-value, traj_y_max])

traj_y_min = 0.3
traj_y_max = 0.5
y_values = np.linspace(traj_y_min, traj_y_max, num=80)
for value in y_values:
    trajectory_points.append([traj_x_min, value])
for value in x_values:
    trajectory_points.append([value, traj_y_max])

traj_y_min = 0.5
traj_y_max = 0.7
y_values = np.linspace(traj_y_min, traj_y_max, num=80)
for value in y_values:
    trajectory_points.append([traj_x_max, value])
for value in x_values:
    trajectory_points.append([1-value, traj_y_max])

traj_y_min = 0.7
traj_y_max = 0.9
y_values = np.linspace(traj_y_min, traj_y_max, num=80)
for value in y_values:
    trajectory_points.append([traj_x_min, value])
for value in x_values:
    trajectory_points.append([value, traj_y_max])

trajectory_array = np.array(trajectory_points)
global mapped_9_before
global mapped_8_before
global mapped_7_before
global mapped_6_before
global mapped_5_before
global mapped_4_before
global mapped_3_before
global mapped_2_before
global mapped_1_before

mapped_9_before = [None, None]
mapped_8_before = [None, None]
mapped_7_before = [None, None]
mapped_6_before = [None, None]
mapped_5_before = [None, None]
mapped_4_before = [None, None]
mapped_3_before = [None, None]
mapped_2_before = [None, None]
mapped_1_before = [None, None]
def calculate_moving_average(mapped_x, mapped_y):
    global mapped_9_before
    global mapped_8_before
    global mapped_7_before
    global mapped_6_before
    global mapped_5_before
    global mapped_4_before
    global mapped_3_before
    global mapped_2_before
    global mapped_1_before

    mapped_total_x, mapped_total_y = mapped_x, mapped_y
    i = 1
    if(mapped_9_before != [None, None]):
        mapped_total_x += mapped_9_before[0]
        mapped_total_y += mapped_9_before[1]
        i += 1
    if(mapped_8_before != [None, None]):
        mapped_total_x += mapped_8_before[0]
        mapped_total_y += mapped_8_before[1]
        mapped_9_before = mapped_8_before
        i += 1
    if(mapped_7_before != [None, None]):
        mapped_total_x += mapped_7_before[0]
        mapped_total_y += mapped_7_before[1]
        mapped_8_before = mapped_7_before
        i += 1
    if(mapped_6_before != [None, None]):
        mapped_total_x += mapped_6_before[0]
        mapped_total_y += mapped_6_before[1]
        mapped_7_before = mapped_6_before
        i += 1
    if(mapped_5_before != [None, None]):
        mapped_total_x += mapped_5_before[0]
        mapped_total_y += mapped_5_before[1]
        mapped_6_before = mapped_5_before
        i += 1
    if(mapped_4_before != [None, None]):
        mapped_total_x += mapped_4_before[0]
        mapped_total_y += mapped_4_before[1]
        mapped_5_before = mapped_4_before
        i += 1
    if(mapped_3_before != [None, None]):
        mapped_total_x += mapped_3_before[0]
        mapped_total_y += mapped_3_before[1]
        mapped_4_before = mapped_3_before
        i += 1
    if(mapped_2_before != [None, None]):
        mapped_total_x += mapped_2_before[0]
        mapped_total_y += mapped_2_before[1]
        mapped_3_before = mapped_2_before
        i += 1
    if(mapped_1_before != [None, None]):
        mapped_total_x += mapped_1_before[0]
        mapped_total_y += mapped_1_before[1]
        mapped_2_before = mapped_1_before
        i += 1
    mapped_1_before = [mapped_x, mapped_y]

    mapped_x, mapped_y = mapped_total_x / i, mapped_total_y / i
    return mapped_x, mapped_y

global prev_left_x, prev_left_y
global prev_right_x, prev_right_y

prev_left_x, prev_left_y = None, None
prev_right_x, prev_right_y = None, None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)
L_side_0 = []
Left_36 = []
Left_pupil = []
Left_39 = []
Right_42 = []
Right_pupil = []
Right_45 = []
R_side_16 = []
Chin_8 = []
Nose_30 = []
Nose_27 = []
Forehead_71 = []

Left_37 = []
Left_38 = []
Left_40 = []
Left_41 = []

Right_43 = []
Right_44 = []
Right_46 = []
Right_47 = []

Nose_28 = []
Nose_29 = []

Chin_7 = []
Chin_9 = []

Mapped = []
stop = False
for i in range(trajectory_array.shape[0]):
    visualisation = np.full((screen_height, screen_width, 3), 255)
    if i == 1:
        key = cv2.waitKey(1000)
    else:
        key = cv2.waitKey(1)
    if key == 27:
        stop = True
        break

    mapped_x, mapped_y = scale_up_values(trajectory_array[i][0], trajectory_array[i][1], screen_width, screen_height)
    
    _ , frame = cap.read()
    try:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        print("Camera in use")
        quit()

    faces = detector(grey)
    for face in faces:
        height, width, _ = frame.shape

        landmarks = predictor(grey, face)

        left = left_eye(frame, landmarks)
        right = right_eye(frame, landmarks)

        if(left.pupilFound and right.pupilFound):
            scaled_left_0_x, scaled_left_0_y = scale_down_values(landmarks.part(0).x, landmarks.part(0).y, width, height)
            scaled_left_36_x, scaled_left_36_y = scale_down_values(landmarks.part(36).x, landmarks.part(36).y, width, height)
            scaled_left_37_x, scaled_left_37_y = scale_down_values(landmarks.part(37).x, landmarks.part(37).y, width, height)
            scaled_left_41_x, scaled_left_41_y = scale_down_values(landmarks.part(41).x, landmarks.part(41).y, width, height)
            scaled_left_x, scaled_left_y = scale_down_values(left.pupil[0], left.pupil[1], width, height)
            scaled_left_38_x, scaled_left_38_y = scale_down_values(landmarks.part(38).x, landmarks.part(38).y, width, height)
            scaled_left_40_x, scaled_left_40_y = scale_down_values(landmarks.part(40).x, landmarks.part(40).y, width, height)
            scaled_left_39_x, scaled_left_39_y = scale_down_values(landmarks.part(39).x, landmarks.part(39).y, width, height)

            scaled_right_42_x, scaled_right_42_y = scale_down_values(landmarks.part(42).x, landmarks.part(42).y, width, height)
            scaled_right_43_x, scaled_right_43_y = scale_down_values(landmarks.part(43).x, landmarks.part(43).y, width, height)
            scaled_right_47_x, scaled_right_47_y = scale_down_values(landmarks.part(47).x, landmarks.part(47).y, width, height)
            scaled_right_x, scaled_right_y = scale_down_values(right.pupil[0], right.pupil[1], width, height)
            scaled_right_44_x, scaled_right_44_y = scale_down_values(landmarks.part(44).x, landmarks.part(44).y, width, height)
            scaled_right_46_x, scaled_right_46_y = scale_down_values(landmarks.part(46).x, landmarks.part(46).y, width, height)
            scaled_right_45_x, scaled_right_45_y = scale_down_values(landmarks.part(45).x, landmarks.part(45).y, width, height)
            scaled_right_16_x, scaled_right_16_y = scale_down_values(landmarks.part(16).x, landmarks.part(16).y, width, height)

            scaled_chin_7_x, scaled_chin_7_y = scale_down_values(landmarks.part(7).x, landmarks.part(7).y, width, height)
            scaled_chin_8_x, scaled_chin_8_y = scale_down_values(landmarks.part(8).x, landmarks.part(8).y, width, height)
            scaled_chin_9_x, scaled_chin_9_y = scale_down_values(landmarks.part(9).x, landmarks.part(9).y, width, height)
            scaled_nose_x, scaled_nose_y = scale_down_values(landmarks.part(30).x, landmarks.part(30).y, width, height)
            scaled_nose_29_x, scaled_nose_29_y = scale_down_values(landmarks.part(29).x, landmarks.part(29).y, width, height)
            scaled_nose_28_x, scaled_nose_28_y = scale_down_values(landmarks.part(28).x, landmarks.part(28).y, width, height)
            scaled_nose_27_x, scaled_nose_27_y = scale_down_values(landmarks.part(27).x, landmarks.part(27).y, width, height)
            scaled_forehead_71_x, scaled_forehead_71_y = scale_down_values(landmarks.part(71).x, landmarks.part(71).y, width, height)

            L_side_0.append((scaled_left_0_x, scaled_left_0_y))
            Left_36.append((scaled_left_36_x, scaled_left_36_y))
            Left_pupil.append((scaled_left_x, scaled_left_y))
            Left_39.append((scaled_left_39_x, scaled_left_39_y))
            Right_42.append((scaled_right_42_x, scaled_right_42_y))
            Right_pupil.append((scaled_right_x, scaled_right_y))
            Right_45.append((scaled_right_45_x, scaled_right_45_y))
            R_side_16.append((scaled_right_16_x, scaled_right_16_y))
            Chin_8.append((scaled_chin_8_x, scaled_chin_8_y))
            Nose_30.append((scaled_nose_x, scaled_nose_y))
            Nose_27.append((scaled_nose_27_x, scaled_nose_27_y))

            Left_37.append((scaled_left_37_x, scaled_left_37_y))
            Left_38.append((scaled_left_38_x, scaled_left_38_x))
            Left_40.append((scaled_left_40_x, scaled_left_40_y))
            Left_41.append((scaled_left_41_x, scaled_left_41_y))

            Right_43.append((scaled_right_43_x, scaled_right_43_y))
            Right_44.append((scaled_right_44_x, scaled_right_44_y))
            Right_46.append((scaled_right_46_x, scaled_right_46_y))
            Right_47.append((scaled_right_47_x, scaled_right_47_y))

            Nose_28.append((scaled_nose_28_x, scaled_nose_28_y))
            Nose_29.append((scaled_nose_29_x, scaled_nose_29_y))

            Chin_7.append((scaled_chin_7_x, scaled_chin_7_y))
            Chin_9.append((scaled_chin_9_x, scaled_chin_9_y))

            Forehead_71.append((scaled_forehead_71_x, scaled_forehead_71_y))

            network_input = np.array([scaled_left_0_x, scaled_left_0_y, scaled_left_36_x, scaled_left_36_y, scaled_left_37_x, scaled_left_37_y, scaled_left_41_x, scaled_left_41_y, scaled_left_x, scaled_left_y,
                                          scaled_left_38_x, scaled_left_38_y, scaled_left_40_x, scaled_left_40_y, scaled_left_39_x, scaled_left_39_y, scaled_right_42_x, scaled_right_42_y, scaled_right_43_x, scaled_right_43_y,
                                          scaled_right_47_x, scaled_right_47_y, scaled_right_x, scaled_right_y, scaled_right_44_x, scaled_right_44_y, scaled_right_46_x, scaled_right_46_y, scaled_right_45_x, scaled_right_45_y,
                                          scaled_right_16_x, scaled_right_16_y, scaled_chin_7_x, scaled_chin_7_y, scaled_chin_8_x, scaled_chin_8_y, scaled_chin_9_x, scaled_chin_9_y, scaled_nose_x, scaled_nose_y,
                                          scaled_nose_29_x, scaled_nose_29_y, scaled_nose_28_x, scaled_nose_28_y, scaled_nose_27_x, scaled_nose_27_y, scaled_forehead_71_x, scaled_forehead_71_y])
            network_input = network_input.reshape(1,48)
            mapping = trained_model.forward_pass(network_input)

            with torch.no_grad():
                mapping = mapping.numpy()
            pred_x, pred_y = calculate_moving_average(mapping[0][0], mapping[0][1])

            Mapped.append((pred_x, pred_y))
        else:
            i -= 1

    if(mapped_x < screen_width and mapped_y < screen_height):
        cv2.circle(visualisation,(mapped_x,mapped_y), 1, (255,0,0), 40)

    visualisation = np.uint8(visualisation)
    cv2.imshow("Calibration", visualisation)
    cv2.setWindowTitle("Calibration", "Calibration point " + str(i))

x = np.array([L_side_0, Left_36, Left_37, Left_41, Left_pupil, Left_38, Left_40, Left_39, Right_42, Right_43, Right_47, Right_pupil, Right_44, Right_46, Right_45, R_side_16,
                 Chin_7, Chin_8, Chin_9, Nose_30, Nose_29, Nose_28, Nose_27, Forehead_71])
x_input = np.swapaxes(x, 0, 1)
x_input = x_input.reshape(x_input.shape[0], 48)
calibration_y = np.array(Mapped)

cap.release()
cv2.destroyAllWindows()

def apply_calibration(array, h):
    array = array.reshape(array.shape[0], 1, 2)
    array = cv2.perspectiveTransform(array, h)
    array = array.reshape(array.shape[0], 2)
    return array

def calculate_magnitude(x1, x2, y1, y2):
    x = abs(x1 - x2)
    y = abs(y1 - y2)
    return ((x ** 2) + (y ** 2)) ** 0.5

def evaluate_metrics(pred_array):
    total_difference = 0
    MSE = torch.mean(torch.square(torch.tensor(pred_array) - torch.tensor(trajectory_array)))
    MAE = torch.mean(abs(torch.tensor(trajectory_array) - torch.tensor(pred_array)))
    for i in range(trajectory_array.shape[0]):
            difference = calculate_magnitude(trajectory_array[i][0], pred_array[i][0], trajectory_array[i][1], pred_array[i][1])
            total_difference += difference
    print(f"Mean scaled Euclidean distance is {total_difference / pred_array.shape[0]}")
    print(f"MSE is {MSE}")
    print(f"MAE is {MAE}")

def evaluate_mm_metrics(pred_array):
    total_difference = 0
    MAE_x = torch.mean(abs(torch.tensor(trajectory_array[0]) - torch.tensor(pred_array[0])))
    MAE_y = torch.mean(abs(torch.tensor(trajectory_array[1]) - torch.tensor(pred_array[1])))
    for i in range(trajectory_array.shape[0]):
            difference = calculate_magnitude(trajectory_array[i][0] * screen_width_mm, pred_array[i][0] * screen_width_mm, trajectory_array[i][1] * screen_height_mm, pred_array[i][1] * screen_height_mm)
            total_difference += difference
    print(f"Mean Euclidean distance for current monitor ({screen_width_mm}x{screen_height_mm}) is {total_difference / pred_array.shape[0]}mm")
    print(f"MAE across x axis of size {screen_width_mm} is {MAE_x * screen_width_mm}")
    print(f"MAE across y axis of size {screen_height_mm} is {MAE_y * screen_height_mm}")

if(not stop):
    h, _ = cv2.findHomography(calibration_y, trajectory_array)
    cal_array = apply_calibration(calibration_y, h)

    print("Pre-calibration:")
    evaluate_metrics(calibration_y)
    print("Post-calibration:")
    evaluate_metrics(cal_array)
    np.save("calibration_function.npy", h)
    print("Post-calibration on current monitor:")
    evaluate_mm_metrics(cal_array)
else:
    print("Stopped prematurely")
input("Press any key to Exit")