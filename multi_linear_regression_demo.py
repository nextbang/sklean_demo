#coding=utf-8
'''
多元线性回归算法
@Author nextbang
@Date 2016-11-05
'''

import seaborn as sns
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split  # 这里是引用了交叉验证

def get_data(file_name,x_field_names,y_field_name):
    '''
    Function for getting data
    '''
    data = pd.read_csv(file_name)
    print data
    X = data[x_field_names]
    Y = data[y_field_name]
    #print X
    #print Y
    return data,X,Y

def divide_train_test_data(X,Y):
    '''
    Function for dividing data
    '''
    X_train,X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2) # If you use random_state=some_number, then you can guarantee that your split will be always the same. 
    print X_train
    print X_test
    print Y_train
    print Y_test
    return X_train,X_test,Y_train,Y_test


def show_linear_plot(data,x_field_names,y_field_name):
    '''
    Function for showing plot
    '''
    sns.pairplot(data, x_vars=x_field_names, y_vars=y_field_name, size=7, aspect=0.8, kind='reg')  
    plot.show()


def train_linear_model(X_train,Y_train):
    '''
    Function for Fitting our data to Linear model
    '''
    # Create linear regression object  
    regr = linear_model.LinearRegression()  
    regr.fit(X_train, Y_train)   #train model  
    print regr.intercept_,regr.coef_
    return regr


def predict_value(regr,X_value):
    '''
    Function for predicting value
    '''
    Y_predict = regr.predict(X_value)  
    print Y_predict
    return Y_predict
    

def predict_rmse(Y_predict,Y_test):
    '''
    Function for pridict of 均方根误差(Root Mean Squared Error, RMSE)
    '''
    len_predict = len(Y_predict)
    sum_mean=0  
    for i in range(len_predict):  
        sum_mean+=(Y_predict[i]-Y_test.values[i])**2  
    sum_erro=np.sqrt(sum_mean/len_predict)  
    # calculate RMSE by hand  
    print "RMSE by hand:",sum_erro  


def show_linear_roc(Y_predict,Y_test):
    '''
    ROC
    '''
    plot.figure()  
    plot.plot(range(len(Y_predict)),Y_predict,'b',label="predict")  
    plot.plot(range(len(Y_test)),Y_test,'r',label="test")  
    plot.legend(loc="upper right") #显示图中的标签  
    plot.xlabel("the number of sales")  
    plot.ylabel('value of sales')  
    plot.show()  


if __name__ == '__main__':
    x_field_names = ['TV','Radio','Newspaper']
    y_field_name = 'Sales'
    data,X,Y = get_data('data/multi_linear_regression.csv',x_field_names,y_field_name)
    X_train,X_test,Y_train,Y_test = divide_train_test_data(X,Y)
    #show_linear_plot(data,x_field_names,y_field_name)

    regr = train_linear_model(X_train,Y_train)
    Y_predict = predict_value(regr,X_test)
    predict_rmse(Y_predict,Y_test)
    show_linear_roc(Y_predict,Y_test)


