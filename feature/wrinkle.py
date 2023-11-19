import math
from PIL import Image
import cv2
import time
import numpy as np
from skimage.filters import frangi, gabor
from skimage import measure, morphology
import dlib


def master_control(g,theta):
    #b, g, r = cv2.split(image)

    if g.shape[0]==0:
        return []


    sk_frangi_img = frangi(g, sigmas=[0,1,0.01], beta=1.5, gamma=0.01)
    sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))


    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=theta)
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(1))


    all_img = cv2.add(0.1 * sk_gabor_img_1, 0.9 * sk_frangi_img)
    all_img = morphology.closing(all_img, morphology.disk(1))

    _, all_img = cv2.threshold(all_img, 0.3, 1, 0)

    bool_img = all_img.astype(bool)
    label_image = measure.label(bool_img)
    count=0

    for region in measure.regionprops(label_image):
        if region.area < 10: #   or region.area > 700
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0

    skel, distance = morphology.medial_axis(all_img.astype(int), return_distance=True)
    skels = morphology.closing(skel, morphology.disk(1))

    return skels


def face_wrinkle(result,theta=0):
    #result = cv2.imread(path)
    img = master_control(result,theta)
    if img==[]:
        ans=0.0
        return [],ans
    count=0
    result[img > 0.1] = 255
    img = img.astype(float)
    for row in img:
        for i in row:
            if not i==0:
                count+=1

    return img,count/(img.shape[0]*img.shape[1])

def get_Wrinkle_vector(face_image_path,detector,predictor):
    img=cv2.imread(face_image_path)
    b, gray, r = cv2.split(img)
    dets = detector(img, 1)
    landmarks=[]
    result=[]
    wri_img=[]
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # Draw the face landmarks on the screen.
        for p in shape.parts():
            landmarks.append([p.x, p.y])
        #print(len(landmarks))
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
        mask = np.zeros(gray.shape[:2], dtype="uint8")
        # mask[left:right,top:bottom]=255
        mask[0:0, 0:0] = 255
        image = cv2.bitwise_and(gray, gray, mask=mask)
        theta=[0,math.pi/2,math.pi/4,math.pi/4*3]
        for i in range(4):
            g=gray
            left=min(x[2*i],x[2*i+1])
            right=max(x[2*i],x[2*i+1])
            top=min(y[2*i],y[2*i+1])
            bottom=max(y[2*i],y[2*i+1])
            mask = np.zeros(g.shape[:2], dtype="uint8")
            mask[top:bottom, left:right] = 255
            g= cv2.bitwise_and(g, g, mask=mask)
            cropped_image=gray[top:bottom, left:right]
            temp_img,temp_res=face_wrinkle(cropped_image,theta=theta[i])
            wri_img.append(temp_img)
            result.append(temp_res)
            image=cv2.add(image,g)
        #cv2.imshow('result', image)
        #cv2.waitKey(0)
        break
    return result

def wrinkle_feature(image_path):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    ################ 0检测横向皱纹   math.pi/2检测竖向皱纹    math.pi/4 | math.pi/4*3 检测斜向皱纹
    result=get_Wrinkle_vector(image_path,detector,predictor)
    if result==[]:
        result=np.zeros(4)
    return result



