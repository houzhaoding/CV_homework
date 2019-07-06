"""
功能：基本图像处理
时间：2019.07.06
作者：houzhaoding
版本：1.0

"""
import cv2
import random
from matplotlib import pyplot as plt
import numpy as np

# change color
#利用产生的随机数对BGR三个通道分别操作,产生不同颜色的图像
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))#通道合并
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
#LUT(参数1，参数2，参数3)函数，将某一源图像的值，按照某一个映射关系table，转变为输出图像的值
    return cv2.LUT(image, table)

def colorchange(image):
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # only for 1 channel
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)  # y: luminance(������), u&v: ɫ�ȱ��Ͷ�
    cv2.imshow('Color input image', image)
    cv2.imshow('Histogram equalized', img_output)
    key = cv2.waitKey(0)
    if key == 27:
        exit()
def image_rotation(image):
    M=cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),0,1)
    print(M)
    img_rotation=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]*1))
    cv2.imshow("lena rotation",img_rotation)
    key = cv2.waitKey(0)
    if key == 27:
        exit()
def image_translate(image):
    TranslationMatrix=np.array([[1,0,100],
                               [0,1,0]],dtype=np.float32)
    img_translate=cv2.warpAffine(image,TranslationMatrix,(image.shape[1],image.shape[0]))
    cv2.imshow("lena translate",img_translate)
    key = cv2.waitKey(0)
    if key == 27:
        exit()
# 仿射变换,保留了线的“直线性”和“平行性”
# 需要取三个点进行对应（2D）
def image_Affine_Transform(image):
    rows, cols, ch = image.shape
    print(type(rows))
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imshow('affine lenna', dst)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
# perspective transform
# 需要取四个点进行对应（3D）
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp


def main():
    img = cv2.imread("E:\AI_training\CV\Week_1\Practice/lena.jpg")
     #img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[1] ), int(img_brighter.shape[0] * 0.5)))
    #输出(256, 512, 3)
    M_warp, img_warp = random_warp(img , img.shape[0], img.shape[1])
    cv2.imshow('lenna_warp', img_warp)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()




if __name__=="__main__":
    main()

