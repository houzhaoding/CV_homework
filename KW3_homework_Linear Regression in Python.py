"""
功能：利用python_way 完成线性回归（利用随机梯度下降法）
版本：1.0
作者：houzhaoding
日期：2019.07.20
"""

import numpy as np
import random
import matplotlib.pyplot as plt


#获取数据集(列表)
def gen_sample_data( num_sample):
    theta1=random.randint(0,10)+random.random()
    theta0=random.randint(0,5)+random.random()
    x_list=[]
    y_list=[]
    for i in range (num_sample):
        x=random.randint(0,100)*random.random()
        y=theta1 *x + theta0 + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    print(theta1,theta0)
    return x_list,y_list,theta1,theta0

#获取数据集(数组)
def gen_sample_matrix(num_sample):
    x_list, y_list, theta1, theta0=gen_sample_data( num_sample)
    X=np.array(x_list)
    X=np.vstack((np.ones_like(X),X))
    Y=np.array(y_list)
    return X, Y

#根据模型返回预测值
def  inference(X,Theta):
    pre_Y=np.dot(Theta,X)
    return pre_Y

#梯度下降更新theta值,使用矩阵形式计算dtheta时，将不再需要累加
def gradient_step(get_Y,X,Theta,lr):
    pre_Y=inference(X, Theta)
    diff=pre_Y-get_Y
    dtheta=(np.dot(diff,X.T))/X.shape[1]
    Theta-=lr*dtheta
    return Theta

#计算loss值
def eval_loss(X,Theta,get_Y):
    pre_Y=inference(X,Theta)
    diff = pre_Y - get_Y
    loss=np.dot(diff,diff)/2/X.shape[1]
    return loss

#利用随机梯度下降法训练模型
def train(X,get_Y,batch_size,lr,max_iter):
    loss_list=[]
    Theta = [1.0,1.0]
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(X[1]), batch_size)  # 拿batch_size个样本取训练
        batch_x =X[:,batch_idxs ]
        batch_y =get_Y[batch_idxs ]
        Theta=gradient_step(batch_y, batch_x,Theta,lr)
        loss=eval_loss(X,Theta,get_Y)
        loss_list.append(loss)
        print('w:{0}, b:{1}'.format(Theta[1], Theta[0]))
        print('loss is {0}'.format(loss))
    return Theta,loss_list

#图形展示迭代中loss变化，以及样本散点
def show(X,get_Y,loss_list,Theta):
    fig,[plt1,plt2]=plt.subplots(1,2)
    fig.suptitle("Linear Regression")

#绘制样本散点图

    plt1.set_xlabel("X")
    plt1.set_ylabel("Y")
    plt1.scatter(X[1,:], get_Y,label="sample")

#绘制拟合后的直线
    x = np.linspace(0, 100, 100)
    y=Theta[1]*x+Theta[0]
    plt1.plot(x, y, '-r', label='y=wx+b')
    plt1.legend()

#绘制loss变化图
    plt2.plot(range(len(loss_list)),loss_list,label="loss")
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.show()








def run():
    num_sample=100
    lr=0.001
    max_iter=10000
    batch_size=50
    X,get_Y=gen_sample_matrix(num_sample)
    Theta,loss_list = train(X, get_Y, batch_size, lr, max_iter)
    show(X,get_Y,loss_list,Theta)








if __name__=="__main__":
    run()