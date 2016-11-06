#coding=utf-8
'''
逻辑回归模型训练
@Author nextbang
@Date 2016-11-06
'''


from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR

import sklearn_demo.util.regression_util as regr_util  # 数据导入、拆分、绘图等



def get_support_fields(X,Y):
    '''
    Function for getting support fields
    '''
    rlr = RLR() #建立随机逻辑回归模型，筛选变量
    rlr.fit(X, Y) #训练模型
    rlr.get_support() #获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
    print rlr.scores_
    print(u'有效特征为：%s' % (','.join(data.columns[rlr.get_support()])).decode('utf-8'))
    X = data[data.columns[rlr.get_support()]].as_matrix() #筛选好特征
    return X


def train_lr_model(X,Y):
    regr = LR() #建立逻辑回归模型
    regr.fit(X,Y) #用筛选后的特征数据来训练模型
    print(u'模型的平均正确率为：%s' % regr.score(X, Y)) #给出模型的平均正确率，本例为81.4%
    return regr


if __name__ == '__main__':
    # 获取训练数据，并plot数据
    x_field_names = 'age,education,work_age,address,income,credit_card,other_credit'.split(',')
    y_field_name = 'violate_contact'
    data,X,Y = regr_util.get_data('data/logistic_regression.csv',x_field_names,y_field_name)
    #regr_util.show_linear_plot(data,x_field_names,y_field_name)

    # 提取有效特征，并拆分训练数据和测试数据
    X = get_support_fields(X,Y)
    X_train,X_test,Y_train,Y_test = regr_util.divide_train_test_data(X,Y)

    # 训练模型，并预测
    regr = train_lr_model(X,Y)
    Y_predict = regr_util.predict_value(regr,X_test)
    
    

