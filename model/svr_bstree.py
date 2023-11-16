import collections
from model.svr import MYSVR
import math
from feature import feature_extraction
import numpy as np

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
        elements=self.get_not_leaf_node()

        for ele in elements:
            face,age=dataloader.get_train_data(ele.start_age,ele.end_age)
            if face==[]:
                continue
            age = np.array(age).reshape(-1,1)
            feature=[]
            for img in face:
                print(img)
                feature_temp=feature_extraction(img)
                feature.append(feature_temp)

            feature=np.array(feature)
            print(feature)
            ele.decision.train(feature,age)
        return True

    def predict(self,face):
        feature=feature_extraction(face)
        feature=np.array(feature).reshape(1,-1)

        node=self.Root
        while not(node.left==None and node.right==None):
            age=node.decision.forward(feature)
            if age<=node.middle:
                node=node.left
            else:
                node=node.right

        return node.end_age


    def plot_predict(self,face):
        feature=feature_extraction(face)

        node=self.Root
        while not(node.left==None and node.right==None):
            age=node.decision.forward(feature)
            if age<=node.middle:
                node=node.left
            else:
                node=node.right

        return node.end_age