"""
功能：利用python_way 完成逻辑回归（利用随机梯度下降法）
版本：1.0
作者：houzhaoding
日期：2019.08.04
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def gen_sample_data( num_sample):
    theta0=-10+random.random()
    theta1 = 200+ random.random()
    x_list=[]
    y_list=[]
    for i in range (num_sample):
        x=random.randint(0,50)*random.random()
        p=sigmoid(theta0* x+theta1)
        x_list.append(x)
        y_list.append(0 if p<0.5 else 1)
    print("origin_theta",theta0,theta1)

    return x_list,y_list



#获取数据集(数组)
def gen_sample_matrix(num_sample):
    x_list, y_list=gen_sample_data(num_sample)
    X=np.asarray(x_list)
    Y=np.asarray(y_list)
    return X, Y

def sigmoid(Z):

    return(1.0/(1.0+np.exp(-Z)))

#根据模型返回预测值
def  inference(X,Theta):
    p=sigmoid(Theta[0]*X+Theta[1])
    pre_Y=np.asarray([0 if i<0.5 else 1 for i in p])
    return pre_Y

#梯度下降更新theta值,使用矩阵形式计算dtheta时，将不再需要累加
def gradient_step(get_Y,X,Theta):

    pre_Y=inference(X, Theta)
    diff=pre_Y-get_Y
    dtheta0=diff * X
    dtheta1=diff
    return dtheta0,dtheta1

def cal_step_gradient(batch_y, batch_x,Theta,lr):
    batch_size=len(batch_x)
    dtheta0,dtheta1=gradient_step(batch_y,batch_x,Theta)
    avg_dtheta0=sum(dtheta0)/batch_size
    avg_dtheta1=sum(dtheta1)/batch_size
    Theta[0]=Theta[0]-lr*avg_dtheta0
    Theta[1] = Theta[1] - lr * avg_dtheta1
    return Theta

#计算loss值
def eval_loss(X,Theta,get_Y):
    pre_Y=inference(X,Theta)
    loss=-get_Y * np.log(sigmoid(pre_Y )) - (1 - get_Y) * np.log(1 - sigmoid(pre_Y ))
    loss=sum(loss)/len(X)
    return loss

#利用随机梯度下降法训练模型
def train(X,get_Y,batch_size,lr,max_iter):
    loss_list=[]
    Theta = np.zeros(2)
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(X), batch_size)  # 拿batch_size个样本取训练
        batch_x =X[batch_idxs ]
        batch_y =get_Y[batch_idxs ]
        Theta=cal_step_gradient(batch_y, batch_x,Theta,lr)
        loss=eval_loss(X,Theta,get_Y)
        loss_list.append(loss)
        print('theta0={0}, theta1={1}'.format(Theta[0],Theta[1]))
        print('loss is {0}'.format(loss))
    return Theta,loss_list

#图形展示迭代中loss变化，以及样本散点


def show(X,get_Y,loss_list,Theta):
    fig,[plt1,plt2]=plt.subplots(1,2)
    fig.suptitle("logistic Regression")

#绘制样本散点图

    plt1.set_xlabel("X")
    plt1.set_ylabel("Y")
    plt1.scatter(range(len(X)), get_Y,label="sample")

#绘制拟合后的曲线

#绘制loss变化图
    plt2.plot(range(len(loss_list)),loss_list,label="loss")
    plt2.set_xlabel("iter")
    plt2.set_ylabel("loss")
    plt.show()








def run():
    num_sample=10
    lr=0.001
    max_iter=5000
    batch_size=10
    X,get_Y=gen_sample_matrix(num_sample)
    Theta,loss_list = train(X, get_Y, batch_size, lr, max_iter)
    show(X, get_Y, loss_list, Theta)









if __name__=="__main__":
    run()