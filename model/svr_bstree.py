import collections
from model.svr import MYSVR
import math
from feature import feature_extraction
import numpy as np
import joblib


class Node:
    def __init__(self,start_age,end_age):
        self.left = None
        self.right = None
        self.start_age=start_age
        self.end_age=end_age
        self.middle=math.floor((self.start_age+self.end_age)/2)
        self.decision=MYSVR()

class SVR_BSTree:

    def __init__(self,min_age,max_age):
        self.min_age=min_age
        self.max_age=max_age

        self.Root = Node(self.min_age,self.max_age)
        self.delta = 5

    def build_tree(self,node=None):
        if node==None:
            node=self.Root

        start_age=node.start_age
        end_age=node.end_age

        if start_age>end_age:
            return False

        elif start_age==end_age:
            return True

        elif start_age==end_age-1:
            node.left=Node(start_age,start_age)
            node.right=Node(end_age,end_age)
            return True

        else:
            middle=math.floor((start_age+end_age)/2)
            node.left=Node(start_age,middle)
            node.right=Node(middle+1,end_age)
            self.build_tree(node.left)
            self.build_tree(node.right)
            return True

    def show_tree(self,node=None):
        if node==None:
            node=self.Root

        print([node.start_age,node.end_age])

        if not node.left==None:
            self.show_tree(node.left)

        if not node.right==None:
            self.show_tree(node.right)

    def get_not_leaf_node(self,node=None,elements=None):
        if node==None:
            node=self.Root
            elements=[]

        if not (node.left==None and node.right==None):
            elements.append(node)
            if not node.left==None:
                elements=self.get_not_leaf_node(node.left,elements)
            if not node.right==None:
                elements=self.get_not_leaf_node(node.right,elements)

        return elements

    def regression(self,dataloader):
        print('Start regression...')
        elements=self.get_not_leaf_node()

        for ele in elements:
            feature,age=dataloader.get_train_data(ele.start_age-self.delta,ele.end_age+self.delta)
            if feature.shape[0]==0:
                continue
            age = np.array(age).reshape(-1,1)
            ele.decision.train(feature,age)
        print('Done!')
        return True

    def predict(self,dataloader):
        feature, age = dataloader.get_test_data()
        count=0
        err=0

        for i in range(0,len(age)):
            feature_ele=feature[i]
            feature_ele=np.array(feature_ele).reshape(1,-1)
            age_ele=age[i]
            if feature_ele.shape[0]==0:
                continue

            count+=1

            node=self.Root
            while not(node.left==None and node.right==None):
                age_temp=node.decision.forward(feature_ele)
                if age_temp<=node.middle:
                    node=node.left
                else:
                    node=node.right

            err+=math.fabs(age_ele-node.end_age)

        print("平均预测误差：")
        print(err/count)
        return True

    def predict_plot(self, face):
        feature = feature_extraction(face)
        feature = np.array(feature).reshape(1, -1)

        node = self.Root
        while not (node.left == None and node.right == None):
            age = node.decision.forward(feature)
            if age <= node.middle:
                node = node.left
            else:
                node = node.right

        print(node.end_age)

        return True