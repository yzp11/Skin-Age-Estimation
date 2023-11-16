from feature.Skin_feature import skin_feature
from feature.wrinkle import wrinkle_feature
from sklearn.preprocessing import StandardScaler
import numpy as np

def feature_extraction(path):
    x1=skin_feature(path)
    x1=np.array(x1)
    x2=wrinkle_feature(path)
    x2=np.array(x2)

    min=np.min(x1)
    max=np.max(x1)

    if not (min==0 and max==0):
        for i in range(0, len(x1)):
            x1[i]=(x1[i]-min)/(max-min)


    min=np.min(x2)
    max=np.max(x2)

    if not (min==0 and max==0):
        for i in range(0, len(x2)):
            x2[i]=(x2[i]-min)/(max-min)

    return np.concatenate((x1,x2))



