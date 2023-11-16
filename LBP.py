import numpy as np
from skimage.feature import local_binary_pattern
import cv2
def LBP_extract(img_path):
    img=cv2.imread(img_path)
    b, g, r = cv2.split(img)
    # Extract the patterns using lbp
    n_point, radius = 16, 2
    lbp = local_binary_pattern(g, n_point, radius, 'default')
    max_bins = int(lbp.max() + 1)  # .max()取lbp中最大的数
    # test_hist是某个灰度级的个数即y坐标。-是横坐标。
    #test_hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    test_hist, _ = np.histogram(lbp, bins=max_bins, range=(0, max_bins))
    return test_hist





