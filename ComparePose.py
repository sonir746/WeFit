import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import utils

BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
               "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
               ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
               ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

net = cv.dnn.readNetFromCaffe("pose_deploy_linevec_faster_4_stages.prototxt", "pose_iter_160000.caffemodel")

def compare_pose(button,image):
    frame1 = cv.imread("TrainImages\{}.jpg".format(button)) # input
    frame2 = cv.cvtColor(image, cv.COLOR_RGB2BGR) # test

    # Ensure images are the same size
    hi = min(frame1.shape[0], frame2.shape[0])
    wi = min(frame1.shape[1], frame2.shape[1])
    frame1 = cv.resize(frame1, (wi,hi))
    frame2 = cv.resize(frame2, (wi,hi))
    frameWidth = frame1.shape[1]
    frameHeight = frame1.shape[0]
    
    # A default value set 
    inWidth = 300
    inHeight = 300
    
    inp = cv.dnn.blobFromImage(frame1, (1.0 / 255), (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    
    pose1 = []
    conf1 = []
    for i in range(len(BODY_PARTS)):

        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold set to 0.1.
        pose1.append((int(x), int(y)) if conf > 0.1 else None)
        conf1.append(conf)
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if pose1[idFrom] and pose1[idTo]:
            cv.line(frame1, pose1[idFrom], pose1[idTo], (255, 74, 0), 3)
            cv.ellipse(frame1, pose1[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame1, pose1[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame1, str(idFrom), pose1[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame1, str(idTo), pose1[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
        
    inp = cv.dnn.blobFromImage(frame2, (1.0 / 255), (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    
    pose2 = []
    for i in range(len(BODY_PARTS)):

        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        pose2.append((int(x), int(y)) if conf > 0.1 else None)
    
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if pose2[idFrom] and pose2[idTo]:
            cv.line(frame2, pose2[idFrom], pose2[idTo], (255, 74, 0), 3)
            cv.ellipse(frame2, pose2[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.ellipse(frame2, pose2[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv.FILLED)
            cv.putText(frame2, str(idFrom), pose2[idFrom], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
            cv.putText(frame2, str(idTo), pose2[idTo], cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv.LINE_AA)
    
    pose1=[(0,0) if key is None else key for key in pose1]
    pose2=[(0,0) if key is None else key for key in pose2]
    
    def similarity_score(pose1, pose2):
        p1 = []
        p2 = []
        pose_1 = np.array(pose1, dtype=np.float64)
        pose_2 = np.array(pose2, dtype=np.float64)

        # Normalize coordinates
        pose_1[:,0] = pose_1[:,0] / max(pose_1[:,0])
        pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
        pose_2[:,0] = pose_2[:,0] / max(pose_2[:,0])
        pose_2[:,1] = pose_2[:,1] / max(pose_2[:,1])

        # Turn (16x2) into (32x1)
        for joint in range(pose_1.shape[0]):
            x1 = pose_1[joint][0]
            y1 = pose_1[joint][1]
            x2 = pose_2[joint][0]
            y2 = pose_2[joint][1]

            p1.append(x1)
            p1.append(y1)
            p2.append(x2)
            p2.append(y2)

        p1 = np.array(p1)
        p2 = np.array(p2)

        # Looking to minimize the distance if there is a match
        # Computing two different distance metrics
        scoreA = utils.cosine_distance(p1, p2)
        scoreB = utils.weight_distance(p1, p2, conf1)
        
        return [scoreA,scoreB]
    
    scores=similarity_score(pose1, pose2)
    
    if scores[1]<0.15:
        return True
    else:
        return False