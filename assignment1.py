import cv2
import random
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('E:\Learning materials\python\lenna.jpg')
print(img.shape)
height, width, channel = img.shape

#图像裁剪
def img_crop(img, crop_area=[]):
    print("Image Crop.")
    if crop_area == []:
        return  img
    else:
        height, width, channel = img.shape
        img_crop = img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3], :]
        return img_crop

# 颜色变换
def img_color_shift(img,brightness=0):
    print("Image Color Shift")
    # 返回颜色变换后的图片
    B, G, R = cv2.split(img)
    def change_brightness(bright):
        # 1为变亮，2为变暗，其他值亮度不变
        if brightness == 1:
            bright_rand = random.randint(30, 100)
        elif brightness == 2:
            bright_rand = random.randint(-100, -30)
        else:
            bright_rand = 0
        print(bright_rand)
        if bright_rand == 0:
            pass
        elif bright_rand > 0:
            lim = 255-bright_rand
            bright[bright > lim] = 255  # 布尔索引
            bright[bright <= lim] = (bright[bright <= lim] + bright_rand).astype(img.dtype)

        elif bright_rand < 0:
            lim = - bright_rand
            bright[bright < lim] = 0
            bright[bright >= lim] = (bright[bright >= lim] + bright_rand).astype(img.dtype)
    #  B,G,R图层依次处理
    for i in [B, G, R]:
        change_brightness(i)
    img_merge = cv2.merge((B, G, R))  # 三个图层合并
    return img_merge

# 图像旋转
def img_rotation(img, angle=0):

    M = cv2.getRotationMatrix2D((int(img.shape[1]/2), int(img.shape[0]/2)), angle, 1)  # 图像旋转矩阵:图像中心，旋转角度，缩放倍数
    img_rotate = cv2.warpAffine(img, M, (img.shape[1],img.shape[0])) # 旋转后的图像
    return img_rotate


#投影变换
def img_perspective(img, margin = 0):
    # 返回投影变换后的图片
    height, width, channels = img.shape
    # margin = 30 # margin 用于调节变换幅度，越大则变换幅度越大
    # 随机生成变换前的四个点的坐标
    x1 = random.randint(-margin, margin)
    y1 = random.randint(-margin, margin)
    x2 = random.randint(width - margin - 1, width - 1)
    y2 = random.randint(-margin, margin)
    x3 = random.randint(width - margin - 1, width - 1)
    y3 = random.randint(height - margin - 1, height - 1)
    x4 = random.randint(-margin, margin)
    y4 = random.randint(height - margin - 1, height - 1)
    # 随机生成变换后的四个点的坐标
    dx1 = random.randint(-margin, margin)
    dy1 = random.randint(-margin, margin)
    dx2 = random.randint(width - margin - 1, width - 1)
    dy2 = random.randint(-margin, margin)
    dx3 = random.randint(width - margin - 1, width - 1)
    dy3 = random.randint(height - margin - 1, height - 1)
    dx4 = random.randint(-margin, margin)
    dy4 = random.randint(height - margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # 变换前的四个点坐标
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])  # 这四个点变换后的坐标
    M_warp = cv2.getPerspectiveTransform(pts1, pts2) # 变换矩阵
    img_warp = cv2.warpPerspective(img, M_warp, (width, height)) # 变换后的图片
    return img_warp


img_show = img_perspective(img, margin=50)
cv2.imshow('lenna', img_show)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()