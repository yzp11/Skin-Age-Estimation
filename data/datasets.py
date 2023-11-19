import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from feature import feature_extraction


class DataLoader:
    def __init__(self,data_path='data.csv'):
        self.datasets = pd.read_csv(data_path)

        self.face=(self.datasets.iloc[:, 0].values)
        self.age =(self.datasets.iloc[:, 1].values)

        self.face_Train, self.face_Test, self.age_Train, self.age_Test = train_test_split(self.face, self.age, test_size=0.2, random_state=0)

        print('Bulid data cache...')
        self.feature_Train = []
        self.feature_Test=[]

        for img in self.face_Train:
            feature_temp = feature_extraction(img)
            self.feature_Train.append(feature_temp)


        for img in self.face_Test:
            #print(img)
            feature_temp = feature_extraction(img)
            self.feature_Test.append(feature_temp)

        print('Done!')




    def get_train_data(self,start_age=0,end_age=200):
        feature_temp=[]
        age_temp=[]
        for k in range (0,self.age_Train.size):
            if self.age_Train[k]>=start_age and self.age_Train[k]<=end_age:
                if not self.feature_Train[k].shape[0] == 10:
                    continue
                feature_temp.append(self.feature_Train[k])
                age_temp.append(self.age_Train[k])

        feature_temp= np.array(feature_temp)
        return feature_temp, age_temp

    def get_test_data(self):
        #返回list
        return self.feature_Test,self.age_Test

    def get_age_range(self):
        max=min=50
        for k in range (0,self.face_Train.size):
            if self.age_Train[k]<=min:
                min=self.age_Train[k]
            if self.age_Train[k]>=max:
                max=self.age_Train[k]

        return min,max
