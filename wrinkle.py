import math
from PIL import Image
import cv2
import time
import numpy as np
from skimage.filters import frangi, gabor
from skimage import measure, morphology
import dlib


def master_control(image,theta):
    b, g, r = cv2.split(image)

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


def face_wrinkle(path,theta=0):
    result = cv2.imread(path)
    img = master_control(result,theta)
    count=0
    result[img > 0.1] = 255
    img=img.astype(float)
    cv2.imshow('ss',img)
    cv2.waitKey()
    for row in img:
        for i in row:
            if not i==0:
                count+=1

    return count/(img.shape[0]*img.shape[1])




if __name__ == '__main__':
    path = r"aaa.jpg"
    theta=0     ################ 0检测横向皱纹   math.pi/2检测竖向皱纹    math.pi/4 | math.pi/4*3 检测斜向皱纹
    count=face_wrinkle(path,theta)
    print(count)

