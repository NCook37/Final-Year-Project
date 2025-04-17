#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pickle
from pathlib import Path
import screeninfo
import dlib
import torch
import torch.optim as optim
import torch.nn as nn
from eye_tracking import *


path = str(Path.cwd())

face_cascade = cv2.CascadeClassifier(path + "/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(path + "/haarcascade_eye.xml")


# In[2]:



global screen_width_mm
global screen_width
global screen_height

monitor = screeninfo.get_monitors()[0]
screen_width_mm = monitor.width_mm
screen_width = monitor.width
screen_height = monitor.height


# In[12]:


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


# In[13]:


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

    mapped_x, mapped_y = int(mapped_total_x / i), int(mapped_total_y / i)
    return mapped_x, mapped_y


# In[14]:


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
        #[w1, w2, w3, w4, w5, w6] = self.params
        #[bn1, bn2, bn3, bn4, bn5] = self.func
        
        batch_inputs_t = torch.tensor(batch_inputs, dtype=torch.float)  # Makes pytorch array to pytorch tensor.
        
        unary_feature_for_bias = torch.ones(size=(batch_inputs.shape[0], 1)) # [N, 1] column vector.
        x = torch.cat((batch_inputs_t, unary_feature_for_bias), dim=1) # Extra feature=1 for bias.
    
        # Layer 1
        h1_preact = self.layer_1(x)
        h1_act = h1_preact.clamp(min=0)
        h1_act = self.bn1(h1_act)
        # Layer 2 (bottleneck):
        h1_ext = torch.cat((h1_act, unary_feature_for_bias), dim=1)
        h2_preact = self.layer_2(h1_ext)
        h2_act = h2_preact.clamp(min=0)
        h2_act = self.bn2(h2_act)
        # Layer 3:
        h2_ext = torch.cat((h2_act, unary_feature_for_bias), dim=1)
        h3_preact = self.layer_3(h2_ext)
        h3_act = h3_preact.clamp(min=0)
        h3_act = self.bn3(h3_act)
        # Layer 4 (output):
        # Output layer
        h3_ext = torch.cat((h3_act, unary_feature_for_bias), dim=1)
        h4_preact = self.layer_4(h3_ext)
        h4_act = h4_preact.clamp(min=0)
        h4_act = self.bn4(h4_act)

        h4_ext = torch.cat((h4_act, unary_feature_for_bias), dim=1)
        h5_preact = self.layer_5(h4_ext)
        h5_act = h5_preact.clamp(min=0)
        h5_act = self.bn5(h5_act)

        h5_ext = torch.cat((h5_act, unary_feature_for_bias), dim=1)
        h6_preact = self.layer_6(h5_ext)#
        h6_act = h6_preact
        x_pred = h6_act
                
        return x_pred
        
        
def MSE_loss(x_pred, x_real, eps=1e-7):
    # Cross entropy: See Lecture 5, slide 19.
    # x_pred: [N, D_out] Prediction returned by forward_pass. Numpy array of shape [N, D_out]
    # x_real: [N, D_in]
    
    # If number array is given, change it to a Torch tensor.
    x_pred = torch.tensor(x_pred, dtype=torch.float) if type(x_pred) is np.ndarray else x_pred
    x_real = torch.tensor(x_real, dtype=torch.float) if type(x_real) is np.ndarray else x_real

    loss_recon = torch.sqrt(torch.sum(torch.square((x_pred - x_real)), axis= 1))
    
    cost = torch.mean(loss_recon) # Expectation of loss: Mean over samples (axis=0).
    return cost


# In[15]:

trained_model = pickle.load(open(path + '/trained model.sav', 'rb'))
trained_model.eval()

calibration_function = np.load(path + "/calibration_function.npy")

def apply_calibration(array, h):
    array = array.reshape(array.shape[0], 1, 2)
    array = cv2.perspectiveTransform(array, h)
    array = array.reshape(array.shape[0], 2)
    return array


# In[16]:


global prev_left_x, prev_left_y
global prev_right_x, prev_right_y

prev_left_x, prev_left_y = None, None
prev_right_x, prev_right_y = None, None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

###Dlib's pre-trained facial landmark detector is based on a method called the "shape predictor." 
# This predictor takes a bounding box around a detected face as input and outputs a set of 81 (x, y)
# coordinates corresponding to various facial landmarks.
###

monitor = screeninfo.get_monitors()[0]
screen_width_mm = monitor.width_mm
screen_width = monitor.width
screen_height = monitor.height

cap = cv2.VideoCapture(0)
while True:
    if(cap.isOpened):
        _ , frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(grey)
        for face in faces:

            landmarks = predictor(grey, face)

            left = left_eye(frame, landmarks)
            right = right_eye(frame, landmarks)

            if(left.pupilFound and right.pupilFound):
                height, width, _ = frame.shape

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

                network_input = np.array([scaled_left_0_x, scaled_left_0_y, scaled_left_36_x, scaled_left_36_y, scaled_left_37_x, scaled_left_37_y, scaled_left_41_x, scaled_left_41_y, scaled_left_x, scaled_left_y,
                                          scaled_left_38_x, scaled_left_38_y, scaled_left_40_x, scaled_left_40_y, scaled_left_39_x, scaled_left_39_y, scaled_right_42_x, scaled_right_42_y, scaled_right_43_x, scaled_right_43_y,
                                          scaled_right_47_x, scaled_right_47_y, scaled_right_x, scaled_right_y, scaled_right_44_x, scaled_right_44_y, scaled_right_46_x, scaled_right_46_y, scaled_right_45_x, scaled_right_45_y,
                                          scaled_right_16_x, scaled_right_16_y, scaled_chin_7_x, scaled_chin_7_y, scaled_chin_8_x, scaled_chin_8_y, scaled_chin_9_x, scaled_chin_9_y, scaled_nose_x, scaled_nose_y,
                                          scaled_nose_29_x, scaled_nose_29_y, scaled_nose_28_x, scaled_nose_28_y, scaled_nose_27_x, scaled_nose_27_y, scaled_forehead_71_x, scaled_forehead_71_y])
                network_input = network_input.reshape(1,48)
                mapping = trained_model.forward_pass(network_input)
                
                with torch.no_grad():
                    mapping = mapping.numpy()

                mapping = apply_calibration(mapping, calibration_function)
                visualisation = np.full((screen_height, screen_width, 3), 255)


                mapped_x, mapped_y = scale_up_values(mapping[0][0], mapping[0][1], screen_width, screen_height)

                
                mapped_x, mapped_y = calculate_moving_average(mapped_x, mapped_y)

                if(mapped_x < screen_width and mapped_y < screen_height):
                    cv2.circle(visualisation,(mapped_x,mapped_y), 1, (255,0,0), 50)
                    #print(f"Plotted X: {mapped_x} Y: {mapped_y}")

                visualisation = np.uint8(visualisation)

                cv2.imshow("Prediction", visualisation)
            cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
input("Press any key to Exit")
