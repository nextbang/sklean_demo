#coding=utf-8
'''
一元线性回归算法demo
@Author nextbang
@Date 2016-11-05
'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model # 线性回归模型

def get_data(file_name,x_field_name,y_field_name):
    '''
    Function for getting data
    '''
    data = pd.read_csv(file_name)  # here ,use pandas to read cvs file.  sep default ','
    print data
    X = []
    Y = []
    for single_square_feet, single_price_value in zip(data[x_field_name], data[y_field_name]):  # 遍历数据，
        X.append([float(single_square_feet)])  # 存储在相应的list列表中
        Y.append(float(single_price_value))
    return X, Y


def show_linear_plot(X,Y):
    '''
    Function for showing the resutls of linear fit model
    '''
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    plt.scatter(X,Y,color='blue')
    plt.plot(X,regr.predict(X),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def train_linear_model(X,Y):  
    '''
    Function for Fitting our data to Linear model  
    '''
    # Create linear regression object  
    regr = linear_model.LinearRegression()  
    regr.fit(X, Y)   #train model  
    print regr.intercept_,regr.coef_
    return regr


def predict_value(regr,x_value):
    '''
    Function for predicting value
    '''
    predict_value = regr.predict(x_value)  
    return predict_value


if __name__ == '__main__':
    X,Y = get_data('data/linear_regression.csv','housing_area','price')
#    show_linear_plot(X,Y)
    regr=train_linear_model(X,Y)
    print predict_value(regr,90)




