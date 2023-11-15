from PIL import Image
import cv2
import time
import numpy as np
from skimage.filters import frangi, gabor
from skimage import measure, morphology
import dlib
#from Partition import practice14 as pa


def master_control(image):
    # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation=cv2.INTER_CUBIC)  # 图片分辨率很大是要变小
    b, g, r = cv2.split(image)  # image

    detector = dlib.get_frontal_face_detector()
    # 1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
    dets = detector(g, 1)
    print('dets:', dets)  # dets: rectangles[[(118, 139) (304, 325)]]
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        # 人脸的左上和右下角坐标
        left = d.left()
        top = d.top()
        right = d.right()
        bottom = d.bottom()
        mask=np.zeros(g.shape[:2], dtype="uint8")
        mask[left-28:right,top:bottom]=255
        g=cv2.bitwise_and(g,g,mask=mask)

    sk_frangi_img = frangi(g, scale_range=(0, 1), scale_step=0.01, beta=1.5, gamma=0.01)  # 线宽范围，步长，连接程度（越大连接越多），减少程度(越大减得越多)0.015
    sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency=0.35, theta=45)  # 越小越明显
    sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency=0.35, theta=90)
    sk_gabor_img_4, sk_gabor_4 = gabor(g, frequency=0.35, theta=360)  # 横向皱纹
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    #all_img = cv2.add(0.1 * sk_gabor_img_2, 0.9 * sk_frangi_img)  # + 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_2 + 0.02 * sk_gabor_img_3
    all_img = cv2.add( 0.04 *sk_gabor_img_2+ 0.02 * sk_gabor_img_1 + 0.02 * sk_gabor_img_4 + 0.02 * sk_gabor_img_3,0.9* sk_frangi_img)
    all_img = morphology.closing(all_img, morphology.disk(1))
    _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
    img1 = all_img
    # print(all_img, all_img.shape, type(all_img))
    # contours, image_cont = cv2.findContours(all_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # all_img = all_img + image_cont
    bool_img = all_img.astype(bool)
    label_image = measure.label(bool_img)
    count = 0

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
    trans1 = skels  # 细化
    return skels, count  # np.uint16(skels.astype(int))


def face_wrinkle(path):
    # result = pa.curve(path, backage)
    result = cv2.imread(path)
    img, count = master_control(result)
    print(img.astype(float))
    result[img > 0.1] = 255
    cv2.imshow("result", img.astype(float))
    cv2.waitKey(0)


if __name__ == '__main__':
    path = r"aaa.jpg"
    backage = r'..\Sebum\shape_predictor_81_face_landmarks.dat'

    face_wrinkle(path)

