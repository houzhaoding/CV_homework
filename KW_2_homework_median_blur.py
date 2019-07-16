"""
时间：2019.07.15
作者：houzhaoding
版本：1.0
功能：构造中值滤波函数，并考虑不同的padding方式，对原图像进行中值滤波处理
思路：以原输入图像作为循环的中心，利用padding图和原图的位置关系对应像素赋值
"""
import cv2
import numpy as np

def medianBlur(img, kernel,padding_way):
    m,n=kernel.shape
    M,N=img.shape


    img_new=np.zeros((M+m-1,N + n - 1))
#构造增加padding后的图像矩阵
    if padding_way=="ZERO":
        for i in range(M):
            for j in range(N):
                 img_new[i+int((m-1)/2), j+int((n-1)/2)] = img[i, j]

    if padding_way=="REPLICA":
    #为padding四个角点区域赋值
        img_new[0:int((m-1)/2),0:int((n-1)/2)]=img[0,0]
        img_new[M+m-1-int((m-1)/2):M+m-1, 0:int((n - 1) / 2)] = img[M-1, 0]
        img_new[0:int((m - 1) / 2), N+n-1-int((n-1)/2):N + n - 1] = img[0, N-1]
        img_new[M + m - 1 - int((m - 1) / 2):M + m - 1, N+n-1-int((n-1)/2):N + n - 1] = img[M - 1, N-1]
        for i in range(M):
            #左右区域赋值
            img_new[i+int((m - 1) / 2):i + int((m - 1) / 2)+1, 0:int((n - 1) / 2)] = img[i, 0]
            img_new[i+int((m - 1) / 2):i + int((m - 1) / 2)+1, N + n - 1 - int((n - 1) / 2):N + n - 1] = img[i, N - 1]
            for j in range(N):
                 # 中间区域赋值==原图像
                 img_new[i+int((m-1)/2), j+int((n-1)/2)] = img[i, j]
                 # 上下区域赋值
                 img_new[0:int((m-1)/2), j+int((n - 1) / 2):j + int((n - 1) / 2)+1] = img[0, j]
                 img_new[M + m - 1 - int((m - 1) / 2):M + m - 1, j+int((n - 1) / 2):j + int((n - 1) / 2)+1] \
                     = img[M-1, j]
# 循环取中值
    rows,cols=img_new.shape
#构建img_out输出函数，避免直接对原图img操作
    img_out=np.zeros(img.shape,dtype=img.dtype)
    for r in range(M):
        for c in range(N):
#判断上边界
            if r-m<0:
                rTop=0
            else:
                rTop=r-m
#判断下边界
            if r+m>rows-1:
                rBottom=rows-1
            else:
                rBottom=r+m
#判断左边界
            if c-n<0:
                cLeft=0
            else:
                cLeft=c-n
#判断y右边界
            if c+n>cols-1:
                cRight=cols
            else:
                cRight=c+n
#取领域，模板区域的值
            region=img_new[rTop:rBottom+1,cLeft:cRight+1]
#取中值并赋给中间元素
            img_out [r] [c] = np.median(region)
    cv2.imshow('medianBlur', img_out)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()





def main():
    img = cv2.imread('E:\AI_Training\CV\Week_1\Practice\lena.jpg',0)
    cv2.imshow('origin', img)
#输入模板尺寸
    m=3
    n=3
    kernel=np.ones((m,n))
#输入padding的方式
    padding_way="REPLICA"
#调用中值滤波函数
    medianBlur(img, kernel,padding_way)
if __name__=="__main__":
    main()