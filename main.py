from model import SVR_BSTree
from data import DataLoader
from data import data_select

if __name__ =='__main__':

    #data_select('meta.csv')

    data_path='data.csv'
    dataloader=DataLoader(data_path)
    predict_path='aaa.jpg'

    min,max=dataloader.get_age_range()

    my_model=SVR_BSTree(min,max)
    my_model.build_tree()
    my_model.regression(dataloader)

    print( my_model.predict(predict_path) )