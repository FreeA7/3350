import cv2 as cv
import numpy as np
import random
import os

import datetime


def gaussianThreshold(img):
    # 图片进行二值化
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10)
    # cv.namedWindow('GAUSSIAN', cv.WINDOW_AUTOSIZE)
    # cv.imshow('GAUSSIAN', binary)
    return binary


def getModel():
    # 获取模板，返回为一个[q1问题的模板list, q2问题的模板list]
    m1 = cv.imread('./feature/q1/feature5.jpg', cv.IMREAD_GRAYSCALE)
    m2 = cv.imread('./feature/q1/feature6.jpg', cv.IMREAD_GRAYSCALE)
    m3 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)
    m4 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)
    return [[m1, m2], [m3, m4]]


def getMore(img, m):
    # 对目标图片进行所有模板的匹配，返回匹配效果最好的模板以及最大值与位置
    res0 = cv.matchTemplate(img, m[0], cv.TM_CCOEFF_NORMED)
    min_val0, max_val0, min_loc0, max_loc0 = cv.minMaxLoc(res0)

    res1 = cv.matchTemplate(img, m[1], cv.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)

    if max_val0 > max_val1:
        return max_val0, max_loc0, 0
    else:
        return max_val1, max_loc1, 1


def getBest(img, m, q):
    ''' 对目标图片依次进行q1和q2的匹配，
        其中每次匹配都会依次匹配三种不同的分辨率，
        然后得出2x3次匹配中val最大的值
        如果最好val的匹配大于0.75则返回，否则输出error
    '''
    img_t = img.copy()

    max_val_0, max_loc_0, flag_0 = getMore(img, m)

    if not q:
        img = cv.resize(img, (855, 646), cv.INTER_CUBIC)
    else:
        img = cv.resize(img, (863, 647), cv.INTER_CUBIC)

    max_val_1, max_loc_1, flag_1 = getMore(img, m)

    img1 = img_t.copy()
    if not q:
        img1 = cv.resize(img1, (872, 656), cv.INTER_CUBIC)
    else:
        img1 = cv.resize(img1, (876, 657), cv.INTER_CUBIC)

    max_val_2, max_loc_2, flag_2 = getMore(img1, m)

    print('    Flag:')
    print('    0:%f' % (max_val_0))
    print('    1:%f' % (max_val_1))
    print('    2:%f' % (max_val_2))

    list_val = [max_val_0, max_val_1, max_val_2]
    list_loc = [max_loc_0, max_loc_1, max_loc_2]
    list_flag = [flag_0, flag_1, flag_2]
    list_img = [img_t, img, img1]
    i = list_val.index(max(list_val))

    print('    END:%f' % (list_val[i]))

    if list_val[i] >= 0.75:
        return list_val[i], list_loc[i], list_flag[i], list_img[i]
    else:
        return 0, 0, -1, img_t


def getWhich(img, m):
    # 获取目标图片的最好匹配val和位置loc，以及所符合的问题是q1还是q2
    max_val, max_loc, best, img = getBest(img, m[0], 0)
    flag = 0
    if max_val == 0:
        max_val, max_loc, best, img = getBest(img, m[1], 1)
        flag = 1
        if max_val == 0:
            return 0, 0, -1, -1, img
    # flag是指问题是q1还是q2，best是指匹配的最好的模板是这个问题的哪一个模板
    return max_val, max_loc, flag, best, img


def getColor():
    # 画图的时候随机返回一个颜色
    return (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


def getUDMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个水平轴进行对称
    for i in range(len(ptsx)):
        ptsx[i] = np.array(
            [[j[0], ((tl[1] + loc) + ((tl[1] + loc) - j[1]))] for j in ptsx[i]])
    return ptsx


def getLRMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个垂直轴进行对称
    for i in range(len(ptsx)):
        ptsx[i] = np.array(
            [[((tl[0] + loc) + ((tl[0] + loc) - j[0])), j[1]] for j in ptsx[i]])
    return ptsx


def getZeroFlag(pts, img):
    # 判断这个多边形是否有任何一个顶点是在图片的像素范围之内
    flag = 0
    h = img.shape[0]
    w = img.shape[1]
    for i in pts:
        if i[0] > 0 and i[1] > 0 and i[0] < w and i[1] < h:
            flag = 1
            break
    return flag


def getMove(pts, gap, hov):
    # 将列表中的多边形统一向水平或者垂直方向移动gap
    if hov:
        return np.array([[i[0], i[1] + gap] for i in pts])
    else:
        return np.array([[i[0] + gap, i[1]] for i in pts])


def getAllTarget(ptsx, img, gap, hov, offset=0):
    ''' 获取一个多边形list的一个方向（横向或者纵向）上的所有有点的拷贝
        ---------------------------                ---------------------------
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        |           |-|           |       -->      ||-||-||-||-||-||-||-||-| |
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        |                         |                |                         |
        ---------------------------                ---------------------------
    '''

    # 获取ptsx原始拷贝
    ptss = ptsx.copy()

    for pts in ptsx:
        pts = pts.reshape(-1, 1, 2)
        img = cv.polylines(img, [pts], True, (0, 255, 0))

    # 是否改变方向，0为没有1为有
    changeFlag = 0

    # 偏移修正系数
    offset_num = 1

    while 1:

        # 整体移动ptsx
        for i in range(len(ptsx)):
            ptsx[i] = getMove(ptsx[i], gap - offset_num // 6, hov)
        if offset_num > 0:
            offset_num += 1
        elif offset_num < 0:
            offset_num -= 1

        # 定义是否还有多边形存在点，0为没有1为有
        zeroFlag = 0

        # 逐个多边形进行判断并绘点
        for i in range(len(ptsx)):
            # 判断这个多边形是否有点
            if getZeroFlag(ptsx[i], img):
                pts = ptsx[i]
                pts = pts.reshape(-1, 1, 2)
                img = cv.polylines(img, [pts], True, (255, 0, 0))
                # 只要有一个多边形有点就可以继续移动
                zeroFlag = 1

        # 绘点结束，判断是否要进行下一次移动，如果任何多边形有点则继续移动
        if zeroFlag:
            continue
        # 所有多边形没点，并且还未改变过移动方向
        elif not changeFlag:
            # 改变移动方向，ptss回归原始拷贝
            changeFlag = 1
            if offset_num != 0:
                offset_num = -1
            gap = (-1) * gap
            ptsx = ptss
            continue
        # 所有多边形没点并且改变过一次移动方向
        else:
            break
    return img


def get1LR(ptsx, img, hgap, vgap):
    ''' 获取一个多边形list左右一定gap的上下所有多边形
        ---------------------------                ---------------------------
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |           |-|           |       -->      |   |-| hgap |-| hgap |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        |                         |                |   |-|               |-| |
        ---------------------------                ---------------------------
    '''
    ptss = ptsx.copy()
    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1)

    ptsx = ptss

    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], (-1) * hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1)

    return img


def get1UD(ptsx, img, hgap, vgap):
    ''' 获取一个多边形list上下一定gap的左右所有多边形
        --------------------------                 --------------------------
        |                        |                 |                        |
        |                        |                 ||-||-||-||-||-||-||-||-|| 
        |                        |                 |            vgap        |
        |           |-|          |        -->      |            |-|         |
        |                        |                 |            vgap        |
        |                        |                 ||-||-||-||-||-||-||-||-|| 
        |                        |                 |                        | 
        |                        |                 |                        | 
        --------------------------                 --------------------------
    '''
    ptss = ptsx.copy()
    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0)

    ptsx = ptss

    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], (-1) * vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0)

    return img


def get1Target(img, tl, best):
    # 获取所有q1问题的多边形标注
    ptsx = []
    if not best:
        tl = [tl[0] - 503, tl[1] + 62]

    # M1-4
    ptsx.append(np.array([[tl[0] + 133, tl[1] + 12], [tl[0] + 146, tl[1]],
                          [tl[0] + 158, tl[1] + 9], [tl[0] + 313, tl[1] + 9],
                          [tl[0] + 318, tl[1] + 18], [tl[0] + 318, tl[1] + 26],
                          [tl[0] + 308, tl[1] + 16], [tl[0] + 158, tl[1] + 14],
                          [tl[0] + 145, tl[1] + 26]]))

    # # M1-3
    ptsx.append(np.array([[tl[0] + 318, tl[1] + 20], [tl[0] + 338, tl[1] + 20],
                          [tl[0] + 338, tl[1] + 89], [tl[0] + 318, tl[1] + 89]]))

    # M1-2
    ptsx.append(np.array([[tl[0] + 326, tl[1] - 39], [tl[0] + 354, tl[1] - 39],
                          [tl[0] + 354, tl[1] - 68], [tl[0] + 360, tl[1] - 68],
                          [tl[0] + 360, tl[1] - 13], [tl[0] + 381, tl[1] - 13],
                          [tl[0] + 381, tl[1] + 8], [tl[0] + 386, tl[1] + 14],
                          [tl[0] + 386, tl[1] + 29], [tl[0] + 363, tl[1] + 29],
                          [tl[0] + 363, tl[1] + 67], [tl[0] + 389, tl[1] + 67],
                          [tl[0] + 389, tl[1] + 86], [tl[0] + 347, tl[1] + 86],
                          [tl[0] + 347, tl[1] + 13], [tl[0] + 326, tl[1] + 13]]))

    # M1-1-1
    ptsx.append(np.array([[tl[0] + 364, tl[1] - 27], [tl[0] + 371, tl[1] - 33],
                          [tl[0] + 371, tl[1] - 65], [tl[0] + 404, tl[1] - 65],
                          [tl[0] + 404, tl[1] + 90], [tl[0] + 392, tl[1] + 90],
                          [tl[0] + 392, tl[1] + 38], [tl[0] + 386, tl[1] + 35],
                          [tl[0] + 386, tl[1] + 14], [tl[0] + 381, tl[1] + 8],
                          [tl[0] + 381, tl[1] - 13], [tl[0] + 364, tl[1] - 13]]))

    # M1-1-2
    ptsx.append(np.array([[tl[0] + 314, tl[1] - 66], [tl[0] + 314, tl[1] - 58],
                          [tl[0] + 27, tl[1] - 58], [tl[0] + 24, tl[1] - 56],
                          [tl[0] + 24, tl[1] + 74], [tl[0] + 32, tl[1] + 83],
                          [tl[0] + 223, tl[1] + 83], [tl[0] + 225, tl[1] + 79],
                          [tl[0] + 229, tl[1] + 76], [tl[0] + 238, tl[1] + 76],
                          [tl[0] + 242, tl[1] + 80], [tl[0] + 245, tl[1] + 83],
                          [tl[0] + 314, tl[1] + 83], [tl[0] + 314, tl[1] + 90],
                          [tl[0] + 21, tl[1] + 90], [tl[0] + 21, tl[1] - 66]]))

    # M1-1-3
    ptsx.append(np.array([[tl[0] + 21, tl[1] + 90], [tl[0] + 21, tl[1] - 66],
                          [tl[0] - 97, tl[1] - 66], [tl[0] - 97, tl[1] - 58],
                          [tl[0] + 17, tl[1] - 58], [tl[0] + 17, tl[1] + 74],
                          [tl[0] + 10, tl[1] + 83], [tl[0] - 97, tl[1] + 83],
                          [tl[0] - 97, tl[1] + 90]]))

    # M2-1
    ptsx.append(np.array([[tl[0] + 328, tl[1] + 1], [tl[0] + 328, tl[1] - 14],
                          [tl[0] + 331, tl[1] - 18], [tl[0] + 328, tl[1] - 21],
                          [tl[0] + 328, tl[1] - 33], [tl[0] + 336, tl[1] - 33],
                          [tl[0] + 336, tl[1] - 26], [tl[0] + 340, tl[1] - 23],
                          [tl[0] + 344, tl[1] - 26], [tl[0] + 344, tl[1] - 33],
                          [tl[0] + 355, tl[1] - 33], [tl[0] + 360, tl[1] - 38],
                          [tl[0] + 360, tl[1] - 67], [tl[0] + 21, tl[1] - 67],
                          [tl[0] + 21, tl[1] - 79], [tl[0] + 520, tl[1] - 79],
                          [tl[0] + 520, tl[1] - 67], [tl[0] + 368, tl[1] - 67],
                          [tl[0] + 368, tl[1] - 34], [tl[0] + 355, tl[1] - 26],
                          [tl[0] + 350, tl[1] - 26], [tl[0] + 350, tl[1] - 21],
                          [tl[0] + 347, tl[1] - 18], [tl[0] + 350, tl[1] - 13],
                          [tl[0] + 350, tl[1] + 1], [tl[0] + 344, tl[1] + 1],
                          [tl[0] + 344, tl[1] - 11], [tl[0] + 340, tl[1] - 13],
                          [tl[0] + 336, tl[1] - 10], [tl[0] + 336, tl[1] + 1]]))

    # M2-2
    ptsx.append(np.array([[tl[0] + 358, tl[1] - 12], [tl[0] + 380, tl[1] - 12],
                          [tl[0] + 380, tl[1] + 11], [tl[0] + 375, tl[1] + 11],
                          [tl[0] + 375, tl[1] + 40], [tl[0] + 387, tl[1] + 40],
                          [tl[0] + 387, tl[1] + 61], [tl[0] + 364, tl[1] + 61],
                          [tl[0] + 364, tl[1] + 37], [tl[0] + 366, tl[1] + 37],
                          [tl[0] + 366, tl[1] + 34], [tl[0] + 361, tl[1] + 34],
                          [tl[0] + 361, tl[1] + 17], [tl[0] + 366, tl[1] + 17],
                          [tl[0] + 366, tl[1] + 10], [tl[0] + 358, tl[1] + 10]]))

    # M2-3-1
    ptsx.append(np.array([[tl[0] + 318, tl[1] - 64], [tl[0] + 339, tl[1] - 64],
                          [tl[0] + 339, tl[1] - 51], [tl[0] + 344, tl[1] - 46],
                          [tl[0] + 344, tl[1] - 26], [tl[0] + 340, tl[1] - 23],
                          [tl[0] + 336, tl[1] - 26], [tl[0] + 336, tl[1] - 42],
                          [tl[0] + 318, tl[1] - 42]]))

    # M2-3-2
    ptsx.append(np.array([[tl[0] + 344, tl[1] - 11], [tl[0] + 340, tl[1] - 13],
                          [tl[0] + 336, tl[1] - 10], [tl[0] + 336, tl[1] + 12],
                          [tl[0] + 331, tl[1] + 18], [tl[0] + 318, tl[1] + 18],
                          [tl[0] + 318, tl[1] + 39], [tl[0] + 339, tl[1] + 39],
                          [tl[0] + 339, tl[1] + 23], [tl[0] + 344, tl[1] + 14]]))

    # Sub
    ptsx.append(np.array([[tl[0] - 97, tl[1] - 60], [tl[0] - 97, tl[1] + 84],
                          [tl[0] + 26, tl[1] + 84], [tl[0] + 26, tl[1] - 60]]))

    # Main
    ptsx.append(np.array([[tl[0] + 26, tl[1] - 60], [tl[0] + 26, tl[1] + 84],
                          [tl[0] + 318, tl[1] + 84], [tl[0] + 318, tl[1] - 60]]))
    '''
      变换思路：
      1.获取所有纵向图像
      2.获取横向2h距离的所有纵向图像
      3.原始图像进行上下对称
      4.获取对称图像横向h距离的所有纵向图像
    '''
    hgap = 501
    vgap = 167
    ptsx = np.array(ptsx)
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, vgap, 1)

    ptsx = ptss.copy()
    img = get1LR(ptsx, img, 2 * hgap, vgap)

    ptsx = getUDMirror(ptss, tl, 12)

    img = get1LR(ptsx, img, hgap, vgap)

    return img


def get2Target(img, tl, best):
    # 获取所有q2问题的多边形标注
    ptsx = []

    # M1-1
    ptsx.append(np.array([[tl[0] - 14, tl[1] + 25], [tl[0] - 14, tl[1] + 36],
                          [tl[0] - 4, tl[1] + 36], [tl[0] - 3, tl[1] + 48],
                          [tl[0], tl[1] + 50], [tl[0] + 5, tl[1] + 54],
                          [tl[0] + 14, tl[1] + 56], [tl[0] + 23, tl[1] + 54],
                          [tl[0] + 29, tl[1] + 49], [tl[0] + 32, tl[1] + 36],
                          [tl[0] + 71, tl[1] + 36], [tl[0] + 71, tl[1] + 25]]))

    # M1-2
    ptsx.append(np.array([[tl[0] - 4, tl[1] + 63], [tl[0] - 4, tl[1] + 272],
                          [tl[0] + 2, tl[1] + 272], [tl[0] + 2, tl[1] + 163],
                          [tl[0] + 4, tl[1] + 161], [tl[0] + 60, tl[1] + 161],
                          [tl[0] + 62, tl[1] + 163], [tl[0] + 62, tl[1] + 270],
                          [tl[0] + 68, tl[1] + 270], [tl[0] + 68, tl[1] + 45],
                          [tl[0] + 62, tl[1] + 45], [tl[0] + 62, tl[1] + 154],
                          [tl[0] + 61, tl[1] + 155], [tl[0] + 4, tl[1] + 155],
                          [tl[0] + 2, tl[1] + 153], [tl[0] + 2, tl[1] + 63]]))

    # M2-1
    ptsx.append(np.array([[tl[0], tl[1]], [tl[0], tl[1] + 19],
                          [tl[0] + 2, tl[1] + 21], [tl[0] + 8, tl[1] + 21],
                          [tl[0] + 9, tl[1] + 22], [tl[0] + 9, tl[1] + 42],
                          [tl[0] + 12, tl[1] + 44], [tl[0] + 15, tl[1] + 45],
                          [tl[0] + 18, tl[1] + 42], [tl[0] + 18, tl[1] + 21],
                          [tl[0] + 20, tl[1] + 21], [tl[0] + 21, tl[1] + 19],
                          [tl[0] + 21, tl[1] + 1], [tl[0] + 20, tl[1]]]))

    # M2-2
    ptsx.append(np.array([[tl[0] + 6, tl[1] + 32], [tl[0] + 9, tl[1] + 35],
                          [tl[0] + 10, tl[1] + 42], [tl[0] + 12, tl[1] + 44],
                          [tl[0] + 15, tl[1] + 44], [tl[0] + 18, tl[1] + 42],
                          [tl[0] + 19, tl[1] + 34], [tl[0] + 22, tl[1] + 32],
                          [tl[0] + 25, tl[1] + 35], [tl[0] + 25, tl[1] + 46],
                          [tl[0] + 22, tl[1] + 49], [tl[0] + 20, tl[1] + 51],
                          [tl[0] + 15, tl[1] + 52], [tl[0] + 8, tl[1] + 52],
                          [tl[0] + 6, tl[1] + 55], [tl[0] + 2, tl[1] + 57],
                          [tl[0] - 7, tl[1] + 57], [tl[0] - 7, tl[1] + 49],
                          [tl[0] + 4, tl[1] + 49], [tl[0] + 3, tl[1] + 48],
                          [tl[0] + 3, tl[1] + 35]]))

    # M2-3
    ptsx.append(np.array([[tl[0] - 14, tl[1] + 36], [tl[0] - 14, tl[1] + 280],
                          [tl[0] - 5, tl[1] + 280], [tl[0] - 5, tl[1] + 36]]))

    # Main
    ptsx.append(np.array([[tl[0] - 4, tl[1] + 34], [tl[0] - 4, tl[1] + 278],
                          [tl[0] + 68, tl[1] + 278], [tl[0] + 68, tl[1] + 34]]))

    hgap = 85
    vgap = 252
    '''
      变换思路：
      1.获取所有横向图像
      2.获取纵向2v距离的所有横向图像
      3.原始图像进行左右对称
      4.获取对称图像纵向v距离的所有横向图像
    '''
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, hgap, 0)
    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, vgap * 2)
    ptsx = getLRMirror(ptss, tl, 32)
    img = get1UD(ptsx, img, hgap, vgap)
    
    return img


def getCoordinate(img, getimg = 0):
    # 获取图像的所有零件位置

    # 1、保存原始未处理图像便于画图
    oimg = img.copy()

    # 2、或者模板并对图像进行二值化处理
    m = getModel()
    img = gaussianThreshold(img)

    print('进行匹配：')
    # 3、获取是q1还是q2问题以及最好匹配位置
    max_val, max_loc, flag, best, img = getWhich(img, m)

    # 4、如果二值化的图片有过拉伸处理这里对要进行画图的原图也进行同样的处理
    oimg = cv.resize(oimg, (img.shape[1], img.shape[0]), cv.INTER_CUBIC)

    # return 没有一个大于0.75的匹配
    if flag == -1:
        print('Error')
        if not getimg:
            cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
            cv.imshow("match", oimg)
            return 0
        else:
            return [0, oimg]

    # 5、获取最佳位置以及模板大小，并把最佳匹配在图中画出来
    th, tw = m[flag][0].shape[:2]
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    cv.rectangle(oimg, tl, br, (0, 0, 255), 1)

    # 6、根据q1还是q2获取所有零件位置并在图中画出来
    if not flag:
        oimg = get1Target(oimg, tl, best)
    else:
        oimg = get2Target(oimg, tl, best)

    # 7、图片的展示和返回
    if not getimg:
        cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
        cv.imshow("match", oimg)
        return 1
    else:
        return [1, oimg]


def getJPG(path):
    # 返回一个文件夹下所有jpg文件名
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file[-3:].lower() == 'jpg' and not os.path.isdir(file_path):
            list_name.append(file_path)
    return list_name


# start = datetime.datetime.now()
# getCoordinate(cv.imread('./testp/tp/3300_TA881211CE_TAAOLEC0_34_-232.385_-138.598__S_20180816_094924.jpg'))
# end = datetime.datetime.now()
# print('本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))

def main():
    for i in getJPG('./testp/tp/q1/'):
        start = datetime.datetime.now()
        getCoordinate(cv.imread(i))
        end = datetime.datetime.now()
        print('    本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
三种亮度由暗到亮分为0,1,2
标准处理为处理亮度0的图片
*************************
1 --> 0
-------------------
单个p |  w  |  h  |
-------------------
  1   | 262 | 131 |
-------------------
  0   | 292 | 147 |
-------------------

-------------------
 整图 |  w  |  h  |
-------------------
  1   | 855 | 646 |
-------------------
  0   | 768 | 576 |
-------------------

w1->w0:1.11450381
h1->h0:1.12213740

*************************
2 --> 0
-------------------
单个p |  w  |  h  |
-------------------
  2   | 257 | 129 |
-------------------
  0   | 292 | 147 |
-------------------

-------------------
 整图 |  w  |  h  |
-------------------
  2   | 872 | 656 |
-------------------
  0   | 768 | 576 |
-------------------

w2->w0:1.13618677
h2->h0:1.13953488

'''

if __name__ == '__main__':
    main()
