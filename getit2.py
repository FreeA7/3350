import cv2 as cv
import numpy as np
import random
import os

import datetime


def gaussianThreshold(img, showimg=0):
    # 图片进行二值化
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10)
    if showimg:
        cv.namedWindow('GAUSSIAN', cv.WINDOW_AUTOSIZE)
        cv.imshow('GAUSSIAN', binary)
    return binary


def getModel():
    # 获取模板，返回为一个[q1问题的模板list, q2问题的模板list]
    m11 = cv.imread('./feature/q1/feature11.jpg', cv.IMREAD_GRAYSCALE)
    m12 = cv.imread('./feature/q1/feature12.jpg', cv.IMREAD_GRAYSCALE)

    m21 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)
    m22 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)

    return [[m11, m12], [m21, m22]]


def getMore(img, m):
    # 对目标图片进行所有模板的匹配，返回匹配效果最好的模板以及最大值与位置
    res0 = cv.matchTemplate(img, m[0], cv.TM_CCOEFF_NORMED)
    min_val0, max_val0, min_loc0, max_loc0 = cv.minMaxLoc(res0)

    res1 = cv.matchTemplate(img, m[1], cv.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)
    print('A:%f, %f' % (max_val0, max_val1))

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

    if q == 1:
        img = cv.resize(img, (855, 646), cv.INTER_CUBIC)
    elif q == 2:
        img = cv.resize(img, (863, 647), cv.INTER_CUBIC)

    max_val_1, max_loc_1, flag_1 = getMore(img, m)

    img1 = img_t.copy()
    if q == 1:
        img1 = cv.resize(img1, (872, 656), cv.INTER_CUBIC)
    elif q == 2:
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

    if list_val[i] >= 0.5:
        return list_val[i], list_loc[i], list_flag[i], list_img[i]
    else:
        return 0, 0, -1, img_t


def getWhich(img, m, q):
    # 获取目标图片的最好匹配val和位置loc
    max_val, max_loc, best, img = getBest(img, m[q - 1], q)
    if max_val == 0:
        return 0, 0, -1, img
    # best是指匹配的最好的模板是这个问题的哪一个模板
    return max_val, max_loc, best, img


def getColor():
    # 画图的时候随机返回一个颜色
    return (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


def getUDMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个水平轴进行对称
    for key in ptsx.keys():
        ptsx[key] = np.array(
            [[j[0], ((tl[1] + loc) + ((tl[1] + loc) - j[1]))] for j in ptsx[key]])
    return ptsx


def getLRMirror(ptsx, tl, loc):
    # 将列表中的多边形统一以一个垂直轴进行对称
    for key in ptsx.keys():
        ptsx[key] = np.array(
            [[((tl[0] + loc) + ((tl[0] + loc) - j[0])), j[1]] for j in ptsx[key]])
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


def getAllTarget(ptsx, img, gap, hov, ptdic, offset=0):
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

    # 画出初始图像
    for key in ptsx.keys():
        pts = ptsx[key]
        ptdic[key].append(pts.copy())
        pts = pts.reshape(-1, 1, 2)
        img = cv.polylines(img, [pts], True, (0, 255, 0))

    # 是否改变方向，0为没有1为有
    changeFlag = 0

    # 偏移修正系数
    if not offset:
        offset_num = 0
    else:
        offset_num = 1

    while 1:

        # 整体移动ptsx
        for key in ptsx.keys():
            # print(gap - offset_num // (offset + 1))
            ptsx[key] = getMove(
                ptsx[key], gap - offset_num // (offset + 1), hov)
        if offset_num > 0:
            offset_num += 1
        elif offset_num < 0:
            offset_num -= 1

        # 定义是否还有多边形存在点，0为没有1为有
        zeroFlag = 0

        # 逐个多边形进行判断并绘点
        for key in ptsx.keys():
            # 判断这个多边形是否有点
            if getZeroFlag(ptsx[key], img):
                pts = ptsx[key]
                pts = pts.reshape(-1, 1, 2)
                img = cv.polylines(img, [pts], True, (255, 0, 0))
                # 只要有一个多边形有点就可以继续移动
                zeroFlag = 1

        # 绘点结束，判断是否要进行下一次移动，如果任何多边形有点则继续移动
        if zeroFlag:
            for key in ptsx.keys():
                ptdic[key].append(ptsx[key].copy())
            continue
        # 所有多边形没点，并且还未改变过移动方向
        elif not changeFlag:
            # 改变移动方向，ptss回归原始拷贝
            changeFlag = 1
            if offset:
                offset_num = -1
            gap = (-1) * gap
            ptsx = ptss
            continue
        # 所有多边形没点并且改变过一次移动方向
        else:
            break
    return img


def get1LR(ptsx, img, hgap, vgap, ptdic, offset=0):
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
    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1, ptdic, offset)

    ptsx = ptss

    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], (-1) * hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1, ptdic, offset)

    return img


def get1UD(ptsx, img, hgap, vgap, ptdic, offset=0):
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
    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)

    ptsx = ptss

    for key in ptsx.keys():
        ptsx[key] = getMove(ptsx[key], (-1) * vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)

    return img


def get1Target(img, tl, best):
    tl = [tl[0] - 217, tl[1]]
    # 获取所有q1问题的多边形标注
    ptsx = {}
    if not best:
        tl = [tl[0] - 503, tl[1] + 62]

    ptdic = {}

    # M1-4
    ptdic['M1-4'] = []
    ptsx['M1-4'] = np.array([[tl[0] + 133, tl[1] + 12], [tl[0] + 146, tl[1]],
                             [tl[0] + 158, tl[1] + 9], [tl[0] + 313, tl[1] + 9],
                             [tl[0] + 318, tl[1] + 18], [tl[0] + 318, tl[1] + 26],
                             [tl[0] + 308, tl[1] + 16], [tl[0] + 158, tl[1] + 14],
                             [tl[0] + 145, tl[1] + 26]])

    # M1-3
    ptdic['M1-3'] = []
    ptsx['M1-3'] = np.array([[tl[0] + 318, tl[1] + 20], [tl[0] + 338, tl[1] + 20],
                             [tl[0] + 338, tl[1] + 89], [tl[0] + 318, tl[1] + 89]])

    # M1-2
    ptdic['M1-2'] = []
    ptsx['M1-2'] = np.array([[tl[0] + 326, tl[1] - 39], [tl[0] + 354, tl[1] - 39],
                             [tl[0] + 354, tl[1] - 68], [tl[0] + 360, tl[1] - 68],
                             [tl[0] + 360, tl[1] - 13], [tl[0] + 381, tl[1] - 13],
                             [tl[0] + 381, tl[1] + 8], [tl[0] + 386, tl[1] + 14],
                             [tl[0] + 386, tl[1] + 29], [tl[0] + 363, tl[1] + 29],
                             [tl[0] + 363, tl[1] + 67], [tl[0] + 389, tl[1] + 67],
                             [tl[0] + 389, tl[1] + 86], [tl[0] + 347, tl[1] + 86],
                             [tl[0] + 347, tl[1] + 13], [tl[0] + 326, tl[1] + 13]])

    # M1-1-1
    ptdic['M1-1-1'] = []
    ptsx['M1-1-1'] = np.array([[tl[0] + 364, tl[1] - 27], [tl[0] + 371, tl[1] - 33],
                               [tl[0] + 371, tl[1] - 65], [tl[0] + 404, tl[1] - 65],
                               [tl[0] + 404, tl[1] + 90], [tl[0] + 392, tl[1] + 90],
                               [tl[0] + 392, tl[1] + 38], [tl[0] + 386, tl[1] + 35],
                               [tl[0] + 386, tl[1] + 14], [tl[0] + 381, tl[1] + 8],
                               [tl[0] + 381, tl[1] - 13], [tl[0] + 364, tl[1] - 13]])

    # M1-1-2
    ptdic['M1-1-2'] = []
    ptsx['M1-1-2'] = np.array([[tl[0] + 314, tl[1] - 66], [tl[0] + 314, tl[1] - 58],
                               [tl[0] + 27, tl[1] - 58], [tl[0] + 24, tl[1] - 56],
                               [tl[0] + 24, tl[1] + 74], [tl[0] + 32, tl[1] + 83],
                               [tl[0] + 223, tl[1] + 83], [tl[0] + 225, tl[1] + 79],
                               [tl[0] + 229, tl[1] + 76], [tl[0] + 238, tl[1] + 76],
                               [tl[0] + 242, tl[1] + 80], [tl[0] + 245, tl[1] + 83],
                               [tl[0] + 314, tl[1] + 83], [tl[0] + 314, tl[1] + 90],
                               [tl[0] + 21, tl[1] + 90], [tl[0] + 21, tl[1] - 66]])

    # M1-1-3
    ptdic['M1-1-3'] = []
    ptsx['M1-1-3'] = np.array([[tl[0] + 21, tl[1] + 90], [tl[0] + 21, tl[1] - 66],
                               [tl[0] - 97, tl[1] - 66], [tl[0] - 97, tl[1] - 58],
                               [tl[0] + 17, tl[1] - 58], [tl[0] + 17, tl[1] + 74],
                               [tl[0] + 10, tl[1] + 83], [tl[0] - 97, tl[1] + 83],
                               [tl[0] - 97, tl[1] + 90]])

    # M2-1
    ptdic['M2-1'] = []
    ptsx['M2-1'] = np.array([[tl[0] + 328, tl[1] + 1], [tl[0] + 328, tl[1] - 14],
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
                             [tl[0] + 336, tl[1] - 10], [tl[0] + 336, tl[1] + 1]])

    # M2-2
    ptdic['M2-2'] = []
    ptsx['M2-2'] = np.array([[tl[0] + 358, tl[1] - 12], [tl[0] + 380, tl[1] - 12],
                             [tl[0] + 380, tl[1] + 11], [tl[0] + 375, tl[1] + 11],
                             [tl[0] + 375, tl[1] + 40], [tl[0] + 387, tl[1] + 40],
                             [tl[0] + 387, tl[1] + 61], [tl[0] + 364, tl[1] + 61],
                             [tl[0] + 364, tl[1] + 37], [tl[0] + 366, tl[1] + 37],
                             [tl[0] + 366, tl[1] + 34], [tl[0] + 361, tl[1] + 34],
                             [tl[0] + 361, tl[1] + 17], [tl[0] + 366, tl[1] + 17],
                             [tl[0] + 366, tl[1] + 10], [tl[0] + 358, tl[1] + 10]])

    # M2-3-1
    ptdic['M2-3-1'] = []
    ptsx['M2-3-1'] = np.array([[tl[0] + 318, tl[1] - 64], [tl[0] + 339, tl[1] - 64],
                               [tl[0] + 339, tl[1] - 51], [tl[0] + 344, tl[1] - 46],
                               [tl[0] + 344, tl[1] - 26], [tl[0] + 340, tl[1] - 23],
                               [tl[0] + 336, tl[1] - 26], [tl[0] + 336, tl[1] - 42],
                               [tl[0] + 318, tl[1] - 42]])

    # M2-3-2
    ptdic['M2-3-2'] = []
    ptsx['M2-3-2'] = np.array([[tl[0] + 344, tl[1] - 11], [tl[0] + 340, tl[1] - 13],
                               [tl[0] + 336, tl[1] - 10], [tl[0] + 336, tl[1] + 12],
                               [tl[0] + 331, tl[1] + 18], [tl[0] + 318, tl[1] + 18],
                               [tl[0] + 318, tl[1] + 39], [tl[0] + 339, tl[1] + 39],
                               [tl[0] + 339, tl[1] + 23], [tl[0] + 344, tl[1] + 14]])

    # Sub
    ptdic['Sub'] = []
    ptsx['Sub'] = np.array([[tl[0] - 97, tl[1] - 60], [tl[0] - 97, tl[1] + 107],
                            [tl[0] + 26, tl[1] + 107], [tl[0] + 26, tl[1] - 60]])

    # Main
    ptdic['Main'] = []
    ptsx['Main'] = np.array([[tl[0] + 26, tl[1] - 60], [tl[0] + 26, tl[1] + 107],
                             [tl[0] + 404, tl[1] + 107], [tl[0] + 404, tl[1] - 60]])

    hgap = 501
    vgap = 167
    offset = 0
    '''
      变换思路：
      1.获取所有纵向图像
      2.获取横向2h距离的所有纵向图像
      3.原始图像进行上下对称
      4.获取对称图像横向h距离的所有纵向图像
    '''
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, vgap, 1, ptdic, offset)

    ptsx = ptss.copy()
    img = get1LR(ptsx, img, 2 * hgap, vgap, ptdic, offset)

    ptsx = getUDMirror(ptss, tl, 12)

    img = get1LR(ptsx, img, hgap, vgap, ptdic, offset)

    return img, ptdic


def get2Target(img, tl, best):
    # 获取所有q2问题的多边形标注
    ptsx = {}
    ptdic = {}

    # M1-1
    ptdic['M1-1'] = []
    ptsx['M1-1'] = np.array([[tl[0] - 14, tl[1] + 25], [tl[0] - 14, tl[1] + 36],
                             [tl[0] - 4, tl[1] + 36], [tl[0] - 3, tl[1] + 48],
                             [tl[0], tl[1] + 50], [tl[0] + 5, tl[1] + 54],
                             [tl[0] + 14, tl[1] + 56], [tl[0] + 23, tl[1] + 54],
                             [tl[0] + 29, tl[1] + 49], [tl[0] + 32, tl[1] + 36],
                             [tl[0] + 71, tl[1] + 36], [tl[0] + 71, tl[1] + 25]])

    # M1-2
    ptdic['M1-2'] = []
    ptsx['M1-2'] = np.array([[tl[0] - 4, tl[1] + 63], [tl[0] - 4, tl[1] + 272],
                             [tl[0] + 2, tl[1] + 272], [tl[0] + 2, tl[1] + 163],
                             [tl[0] + 4, tl[1] + 161], [tl[0] + 60, tl[1] + 161],
                             [tl[0] + 62, tl[1] + 163], [tl[0] + 62, tl[1] + 270],
                             [tl[0] + 68, tl[1] + 270], [tl[0] + 68, tl[1] + 45],
                             [tl[0] + 62, tl[1] + 45], [tl[0] + 62, tl[1] + 154],
                             [tl[0] + 61, tl[1] + 155], [tl[0] + 4, tl[1] + 155],
                             [tl[0] + 2, tl[1] + 153], [tl[0] + 2, tl[1] + 63]])

    # M2-1
    ptdic['M2-1'] = []
    ptsx['M2-1'] = np.array([[tl[0], tl[1]], [tl[0], tl[1] + 19],
                             [tl[0] + 2, tl[1] + 21], [tl[0] + 8, tl[1] + 21],
                             [tl[0] + 9, tl[1] + 22], [tl[0] + 9, tl[1] + 42],
                             [tl[0] + 12, tl[1] + 44], [tl[0] + 15, tl[1] + 45],
                             [tl[0] + 18, tl[1] + 42], [tl[0] + 18, tl[1] + 21],
                             [tl[0] + 20, tl[1] + 21], [tl[0] + 21, tl[1] + 19],
                             [tl[0] + 21, tl[1] + 1], [tl[0] + 20, tl[1]]])

    # M2-2
    ptdic['M2-2'] = []
    ptsx['M2-2'] = np.array([[tl[0] + 6, tl[1] + 32], [tl[0] + 9, tl[1] + 35],
                             [tl[0] + 10, tl[1] + 42], [tl[0] + 12, tl[1] + 44],
                             [tl[0] + 15, tl[1] + 44], [tl[0] + 18, tl[1] + 42],
                             [tl[0] + 19, tl[1] + 34], [tl[0] + 22, tl[1] + 32],
                             [tl[0] + 25, tl[1] + 35], [tl[0] + 25, tl[1] + 46],
                             [tl[0] + 22, tl[1] + 49], [tl[0] + 20, tl[1] + 51],
                             [tl[0] + 15, tl[1] + 52], [tl[0] + 8, tl[1] + 52],
                             [tl[0] + 6, tl[1] + 55], [tl[0] + 2, tl[1] + 57],
                             [tl[0] - 7, tl[1] + 57], [tl[0] - 7, tl[1] + 49],
                             [tl[0] + 4, tl[1] + 49], [tl[0] + 3, tl[1] + 48],
                             [tl[0] + 3, tl[1] + 35]])

    # M2-3
    ptdic['M2-3'] = []
    ptsx['M2-3'] = np.array([[tl[0] - 14, tl[1] + 36], [tl[0] - 14, tl[1] + 280],
                             [tl[0] - 5, tl[1] + 280], [tl[0] - 5, tl[1] + 36]])

    # Main
    ptdic['Main'] = []
    ptsx['Main'] = np.array([[tl[0] - 18, tl[1] + 34], [tl[0] - 18, tl[1] + 289],
                             [tl[0] + 68, tl[1] + 289], [tl[0] + 68, tl[1] + 34]])

    hgap = 86
    vgap = 255
    offset = 0
    '''
      变换思路：
      1.获取所有横向图像
      2.获取纵向2v距离的所有横向图像
      3.原始图像进行左右对称
      4.获取对称图像纵向v距离的所有横向图像
    '''
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, hgap, 0, ptdic, offset)
    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, vgap * 2, ptdic, offset)
    ptsx = getLRMirror(ptss, tl, 32)
    img = get1UD(ptsx, img, hgap, vgap, ptdic, offset)

    return img, ptdic


def getCoordinate(img, q, getimg=0, showimg=0):
    # 获取图像的所有零件位置

    # 1、保存原始未处理图像便于画图
    oimg = img.copy()

    # 2、或者模板并对图像进行二值化处理
    m = getModel()
    img = gaussianThreshold(img)

    print('进行匹配：')
    # 3、获取是q1还是q2问题以及最好匹配位置
    max_val, max_loc, best, img = getWhich(img, m, q)

    # return 没有一个大于0.75的匹配
    if best == -1:
        print('Error')
        if showimg:
            cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
            cv.imshow("match", oimg)
        if not getimg:
            return [0, 0]
        else:
            return [0, oimg]

    # 4、如果二值化的图片有过拉伸处理这里对要进行画图的原图也进行同样的处理
    oimg = cv.resize(oimg, (img.shape[1], img.shape[0]), cv.INTER_CUBIC)

    # 5、获取最佳位置以及模板大小，并把最佳匹配在图中画出来
    th, tw = m[q - 1][0].shape[:2]
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    cv.rectangle(oimg, tl, br, (0, 0, 255), 1)

    # 6、根据q1还是q2获取所有零件位置并在图中画出来
    if q == 1:
        oimg, ptdic = get1Target(oimg, tl, best)
    elif q == 2:
        oimg, ptdic = get2Target(oimg, tl, best)

    # 7、图片的展示和返回
    if showimg:
        cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
        cv.imshow("match", oimg)
    if not getimg:
        # 返回多边形dic、图片形状、q1还是q2
        return [ptdic, oimg.shape]
    else:
        return [1, oimg]


def getOverlapping(ptss, target, shape):
    im1 = np.zeros(shape, dtype=np.uint8)
    for pts in ptss:
        im1 = cv.fillConvexPoly(im1, pts, 1)

    im2 = np.zeros(shape, dtype=np.uint8)
    target = cv.fillConvexPoly(im2, target, 1)

    # target = cv.resize(target, shape, cv.INTER_CUBIC)
    # target = target // 255

    img = im1 + target

    if (img > 1).any():
        return 1
    else:
        return 0


def getReturn(m1, m2, m):
    result = {}
    result['AFFECTEDPIXELNUM'] = m
    if not m1:
        result['ISM1OPEN'] = False
        result['M1M1'] = False
    else:
        result['ISM1OPEN'] = True
        if m1 > 1:
            result['M1M1'] = True
        else:
            result['M1M1'] = False
    if not m2:
        result['ISM2OPEN'] = False
        result['M2M2'] = False
    else:
        result['ISM2OPEN'] = True
        if m2 > 1:
            result['M2M2'] = True
        else:
            result['M2M2'] = False
    if m1 > 0 and m2 > 0:
        result['M1M2'] = True
    else:
        result['M1M2'] = False
    return result


def getQ1Out(ptdic, target, shape):
    sum_m1 = 0
    sum_m2 = 0
    sum_m = 0

    for i in range(len(ptdic['Main'])):
        m1 = []
        for key in ['M1-1-1', 'M1-1-2', 'M1-1-3', 'M1-2', 'M1-3', 'M1-4']:
            m1.append(ptdic[key][i])
        if getOverlapping(m1, target, shape):
            sum_m1 += 1
            if sum_m1 > 1:
                break

    for i in range(len(ptdic['Main'])):
        m2 = []
        for key in ['M2-1', 'M2-2', 'M2-3-1', 'M2-3-2']:
            m2.append(ptdic[key][i])
        if getOverlapping(m2, target, shape):
            sum_m2 += 1
            if sum_m2 > 1:
                break

    for i in range(len(ptdic['Main'])):
        m = []
        for key in ['Main', 'Sub']:
            m.append(ptdic[key][i])
        if getOverlapping(m, target, shape):
            sum_m += 1

    return getReturn(sum_m1, sum_m2, sum_m)


def getQ2Out(ptdic, target, shape):
    sum_m1 = 0
    sum_m2 = 0
    sum_m = 0

    for i in range(len(ptdic['Main'])):
        m1 = []
        for key in ['M1-1', 'M1-2']:
            m1.append(ptdic[key][i])
        if getOverlapping(m1, target, shape):
            sum_m1 += 1

    for i in range(len(ptdic['Main'])):
        m2 = []
        for key in ['M2-1', 'M2-2', 'M2-3']:
            m2.append(ptdic[key][i])
        if getOverlapping(m2, target, shape):
            sum_m2 += 1

    for i in range(len(ptdic['Main'])):
        m = []
        for key in ['Main']:
            m.append(ptdic[key][i])
        if getOverlapping(m, target, shape):
            sum_m += 1

    return getReturn(sum_m1, sum_m2, sum_m)


def getJPG(path, li=0):
    # 返回一个文件夹下所有jpg文件名
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file[-3:].lower() == 'jpg' and not os.path.isdir(file_path):
            if not li:
                list_name.append(file_path)
            else:
                list_name.append([path, file])
        if os.path.isdir(file_path):
            list_name += getJPG(file_path, li)
    return list_name


# -------------------------- 处理指定path下所有图片并展示 --------------------------
# path = './sampTestPic/'
path = './testp/tp/q1/'
for i in getJPG(path):
    start = datetime.datetime.now()
    [dic, shape] = getCoordinate(cv.imread(i), q=1, showimg=1)
    end = datetime.datetime.now()
    print(i)
    print('    本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))
    cv.waitKey(0)
    cv.destroyAllWindows()

cv.waitKey(0)
cv.destroyAllWindows()


# -------------------------- 处理指定图片并展示 --------------------------
# img = './testp/tp/q1_ERROR/3300_TA881087BC_TAAOLCC0_9_-376.521_-1030.05__S_20180815_201324.jpg'
# start = datetime.datetime.now()
# [dic, shape] = getCoordinate(cv.imread(img), q=1, showimg=1)
# # out = getCoordinate(cv.imread(img), q=1, showimg=1, getimg=1)
# # cv.imwrite('./testp/tp/3300_TA881135AR_TAAOLCC0_15_-1061.61_-1003.41__M_20180815_182454_1.jpg', out[1])
# end = datetime.datetime.now()
# print('本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))

# cv.waitKey(0)
# cv.destroyAllWindows()


# -------------------------- 输出单个重叠情况的测试 --------------------------
# img = './testp/tp/q1/1.jpg'
# start = datetime.datetime.now()
# [dic, shape] = getCoordinate(cv.imread(img), q = 1, showimg=1)

# target = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

# re = getQ1Out(dic, target, shape)

# end = datetime.datetime.now()
# print('单次比较费时%fs:' % (((end - start).microseconds) / 1e6))
# print(re)

# cv.waitKey(0)
# cv.destroyAllWindows()


# -------------------------- 大量测试输出重叠情况的速度^~^ --------------------------
# start = datetime.datetime.now()
# s = 0

# for img in getJPG('./testp/tp/q2/'):
#     s += 1
#     [dic, shape] = getCoordinate(cv.imread(img), q=2)
#     if not dic:
#         continue
#     target = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
#     re = getQ2Out(dic, target, shape)
#     print(re)

# end = datetime.datetime.now()
# all_time = (end - start).seconds + (((end - start).microseconds) / 1e6)
# one_time = all_time / s

# print('共处理%d张图片，共费时%fs，平均每张图片费时%fs' % (s, all_time, one_time))


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
