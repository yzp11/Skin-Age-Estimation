import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self,data_path='data.csv'):
        self.datasets = pd.read_csv(data_path)

        self.face=(self.datasets.iloc[:, 0].values)
        self.age =(self.datasets.iloc[:, 1].values)
        self.face_Train, self.face_Test, self.age_Train, self.age_Test = train_test_split(self.face, self.age, test_size=0.2, random_state=0)

    def get_train_data(self,start_age=0,end_age=99):
        return self.face_Train,self.age_Train

    def get_test_data(self):
        return self.face_Test,self.age_Test
