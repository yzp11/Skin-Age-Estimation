# coding=utf-8
"""
Copyright (c) 2018-2022. All Rights Reserved.

@author: shliang
@email: shliang0603@gmail.com
"""
import numpy as np
import dlib
import cv2
# 下载人脸关键点检测模型： http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_path = "C:/Users/wry/Desktop/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
face_image_path = "C:/Users/wry/Desktop/123.jpg"
img=cv2.imread(face_image_path)
b, g, r = cv2.split(img)
img = dlib.load_rgb_image(face_image_path)
# win.clear_overlay()
# win.set_image(img)
dets = detector(img, 1)
landmarks=[]
print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    # Draw the face landmarks on the screen.
    for p in shape.parts():
        landmarks.append([p.x, p.y])
    print(len(landmarks))
    x1=landmarks[18][0];y1=landmarks[24][1]-landmarks[27][1]+landmarks[30][1]
    x2=landmarks[25][0];y2=landmarks[25][1];y1=2*y2-y1###额头皱纹;
    x3=landmarks[21][0];y3=landmarks[21][1]*2-landmarks[25][1]
    x4=landmarks[22][0];y4=landmarks[25][1]###眉毛间
    x5=landmarks[2][0];y5=landmarks[2][1]
    x6=landmarks[39][0];y6=landmarks[41][1]*2-landmarks[39][1]###左眼眼下
    x7=landmarks[42][0];y7=landmarks[46][1]*2-landmarks[44][1]
    x8=landmarks[14][0];y8=landmarks[14][1]###右眼眼下
    x9=landmarks[21][0];y9=landmarks[27][1]
    x10=landmarks[22][0];y10=landmarks[30][1]###鼻子
    x=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]
    y=[y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]
    for i in range(1):
        left=min(x[2*i],x[2*i+1])
        right=max(x[2*i],x[2*i+1])
        top=min(y[2*i],y[2*i+1])
        bottom=max(y[2*i],y[2*i+1])
        print(left,right,top,bottom)
        mask = np.zeros(g.shape[:2], dtype="uint8")
        #mask[left:right,top:bottom]=255
        mask[top:bottom, left:right] = 255
        print(mask)
        g = cv2.bitwise_and(g, g, mask=mask)
cv2.imshow('result',g)
cv2.waitKey(0)
# win.add_overlay(shape)
# dlib.hit_enter_to_continue()

