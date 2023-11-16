# coding=utf-8
"""
Copyright (c) 2018-2022. All Rights Reserved.

@author: shliang
@email: shliang0603@gmail.com
"""
import numpy as np
import dlib
import cv2
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
def LBP_extract(g):
    # Extract the patterns using lbp
    n_point, radius = 16, 2
    lbp = local_binary_pattern(g, n_point, radius, 'default')
    max_bins = int(lbp.max() + 1)  # .max()取lbp中最大的数
    # test_hist是某个灰度级的个数即y坐标。-是横坐标。
    #test_hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    test_hist, _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins))

    return test_hist


def get_LBP_vector(face_image_path):
    img=cv2.imread(face_image_path)
    b, gray, r = cv2.split(img)
    dets = detector(img, 1)
    landmarks=[]
    result=[]
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
        x=[x5,x6,x7,x8,x9,x10]
        y=[y5,y6,y7,y8,y9,y10]
        mask = np.zeros(gray.shape[:2], dtype="uint8")
        # mask[left:right,top:bottom]=255
        mask[0:0, 0:0] = 255
        image = cv2.bitwise_and(gray, gray, mask=mask)
        for i in range(3):
            g=gray
            left=min(x[2*i],x[2*i+1])
            right=max(x[2*i],x[2*i+1])
            top=min(y[2*i],y[2*i+1])
            bottom=max(y[2*i],y[2*i+1])
            mask = np.zeros(g.shape[:2], dtype="uint8")
            mask[top:bottom, left:right] = 255
            g= cv2.bitwise_and(g, g, mask=mask)
            cropped_image=gray[top:bottom, left:right]
            result.append(LBP_extract(cropped_image))
            image=cv2.add(image,g)
        # cv2.imshow('result', image)
        # cv2.waitKey(0)

    return result


if __name__ == '__main__':
    predictor_path = "C:/Users/wry/Desktop/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    image_path = "C:/Users/wry/Desktop/3.jpg"
    LBP_vector=get_LBP_vector(image_path)
    print(LBP_vector)

    pca = PCA(n_components=2)
    # x=x.reshape(-1,1)
    # 对数据进行降维
    pca.fit(LBP_vector)
    X_pca = pca.fit_transform(LBP_vector)
    print(X_pca)