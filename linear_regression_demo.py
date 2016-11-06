#coding=utf-8
'''
一元线性回归算法demo
@Author nextbang
@Date 2016-11-05
'''

from sklearn import linear_model # 线性回归模型

import sklearn_demo.util.regression_util as regr_util  # 数据导入、拆分、绘图等


def train_linear_model(X,Y):  
    '''
    Function for Fitting our data to Linear model  
    '''
    # Create linear regression object  
    regr = linear_model.LinearRegression()  
    regr.fit(X, Y)   #train model  
    print regr.intercept_,regr.coef_
    return regr


if __name__ == '__main__':

    # 获取训练数据，并plot数据
    x_field_names = 'housing_area'
    y_field_name = 'price'
    data,X,Y = regr_util.get_data('data/linear_regression.csv',x_field_names,y_field_name)
    regr_util.show_linear_plot(data,x_field_names,y_field_name)

    # 拆分训练数据和测试数据
    X_train,X_test,Y_train,Y_test = regr_util.divide_train_test_data(X,Y)

    # 训练模型，并预测
    regr = train_linear_model(X_train.values.reshape(len(X_train),1),Y_train)
    Y_predict = regr_util.predict_value(regr,X_test.values.reshape(len(X_test),1))
    regr_util.predict_rmse(Y_predict,Y_test)
    #regr_util.show_linear_roc(Y_predict,Y_test)  


