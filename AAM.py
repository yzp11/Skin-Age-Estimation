import cv2
import numpy as np
from pyaa import AAM
from skimage import feature


def extract_face_texture_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化AAM模型，使用人脸模型作为形状模型，使用图像作为纹理模型
    aam = AAM(shape_model='face', texture_model='image')

    # 使用AAM模型对图像进行拟合，得到拟合后的形状和纹理模型
    aam.fit(gray)

    # 提取纹理特征向量，使用AAM模型的纹理模型
    texture_features = aam.texture_model.extract_features(gray)
    hist = feature.local_binary_pattern(texture_features, aam.shape_model.n_landmarks, 8, 'uniform')
    hist = np.sum(hist, axis=0) / hist.sum()

    # 提取面部光泽度和平整度特征向量
    shape_features = aam.shape_model.extract_features(gray)
    # 计算面部光泽度特征向量
    glossiness_features = feature.local_binary_pattern(shape_features, aam.shape_model.n_landmarks, 8, 'uniform')
    glossiness_features = np.sum(glossiness_features, axis=0) / glossiness_features.sum()
    # 计算平整度特征向量
    smoothness_features = feature.local_binary_pattern(shape_features, aam.shape_model.n_landmarks, 8, 'uniform')
    smoothness_features = np.sum(smoothness_features, axis=0) / smoothness_features.sum()

    # 返回特征向量
    return np.concatenate([hist.flatten(), glossiness_features.flatten(), smoothness_features.flatten()])