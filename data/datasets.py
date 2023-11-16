import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self,data_path='data.csv'):
        self.datasets = pd.read_csv(data_path)

        self.face=(self.datasets.iloc[:, 0].values)
        self.age =(self.datasets.iloc[:, 1].values)

        self.face_Train, self.face_Test, self.age_Train, self.age_Test = train_test_split(self.face, self.age, test_size=0.2, random_state=0)


    def get_train_data(self,start_age=0,end_age=200):
        face_temp=[]
        age_temp=[]
        for k in range (0,self.face_Train.size):
            if self.age_Train[k]>=start_age and self.age_Train[k]<=end_age:
                face_temp.append(self.face_Train[k])
                age_temp.append(self.age_Train[k])
        return face_temp,age_temp

    def get_test_data(self):
        return self.face_Test,self.age_Test

    def get_age_range(self):
        max=min=50
        for k in range (0,self.face_Train.size):
            if self.age_Train[k]<=min:
                min=self.age_Train[k]
            if self.age_Train[k]>=max:
                max=self.age_Train[k]

        return min,max
