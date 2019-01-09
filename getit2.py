import cv2 as cv
import numpy as np
import random
import os

import datetime


def gaussianThreshold(img):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 10)
    # cv.namedWindow('GAUSSIAN', cv.WINDOW_AUTOSIZE)
    # cv.imshow('GAUSSIAN', binary)
    return binary


def getModel():
    m1 = cv.imread('./feature/q1/feature5.jpg', cv.IMREAD_GRAYSCALE)
    m2 = cv.imread('./feature/q1/feature6.jpg', cv.IMREAD_GRAYSCALE)
    m3 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)
    m4 = cv.imread('./feature/q2/feature3.jpg', cv.IMREAD_GRAYSCALE)
    return [[m1, m2], [m3, m4]]


def getMore(img, m):
    res0 = cv.matchTemplate(img, m[0], cv.TM_CCOEFF_NORMED)
    min_val0, max_val0, min_loc0, max_loc0 = cv.minMaxLoc(res0)

    res1 = cv.matchTemplate(img, m[1], cv.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv.minMaxLoc(res1)

    if max_val0 > max_val1:
        return max_val0, max_loc0, 0
    else:
        return max_val1, max_loc1, 1


def getBest(img, m, q):
    max_val0 = 0
    max_val1 = 0
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
    print('    0:%f' %(max_val_0))
    print('    1:%f' %(max_val_1))
    print('    2:%f' %(max_val_2))

    list_val = [max_val_0, max_val_1, max_val_2]
    list_loc = [max_loc_0, max_loc_1, max_loc_2]
    list_flag = [flag_0, flag_1, flag_2]
    list_img = [img_t, img, img1]
    i = list_val.index(max(list_val))

    print('    END:%f' %(list_val[i]))

    if list_val[i] >= 0.75:
        return list_val[i], list_loc[i], list_flag[i], list_img[i]
    else:
        return 0, 0, -1, img_t


def getWhich(img, m):
    max_val, max_loc, best, img = getBest(img, m[0], 0)
    flag = 0
    if max_val == 0:
        max_val, max_loc, best, img = getBest(img, m[1], 1)
        flag = 1
        if max_val == 0:
            return 0, 0, -1, -1, img
    return max_val, max_loc, flag, best, img


def getColor():
    return (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


def getUDMirror(ptsx, tl, loc):
    for i in range(len(ptsx)):
        ptsx[i] = np.array(
            [[j[0], ((tl[1] + loc) + ((tl[1] + loc) - j[1]))] for j in ptsx[i]])
    return ptsx


def getLRMirror(ptsx, tl, loc):
    for i in range(len(ptsx)):
        ptsx[i] = np.array(
            [[((tl[0] + loc) + ((tl[0] + loc) - j[0])), j[1]] for j in ptsx[i]])
    return ptsx


def getZeroFlag(pts, img):
    flag = 0
    h = img.shape[0]
    w = img.shape[1]
    for i in pts:
        if i[0] > 0 and i[1] > 0 and i[0] < w and i[1] < h:
            flag = 1
            break
    return flag


def getMove(pts, gap, hov):
    if hov:
        return np.array([[i[0], i[1] + gap] for i in pts])
    else:
        return np.array([[i[0] + gap, i[1]] for i in pts])


def getAllTarget(ptsx, img, gap, hov):

    # 获取ptsx原始拷贝
    ptss = ptsx.copy()

    for pts in ptsx:
        pts = pts.reshape(-1, 1, 2)
        img = cv.polylines(img, [pts], True, (0, 255, 0))

    # 是否改变方向，0为没有1为有
    changeFlag = 0

    while 1:
        # 整体移动ptsx
        for i in range(len(ptsx)):
            ptsx[i] = getMove(ptsx[i], gap, hov)

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
            gap = (-1) * gap
            ptsx = ptss
            continue
        # 所有多边形没点并且改变过一次移动方向
        else:
            break
    return img


# def getAllTarget(ptsx, img, gap, hov):
#     for pts in ptsx:
#         ptss = [pts]
#         downflag = 1
#         changeFlag = 0
#         while 1:
#             temp = getMove(ptss[-1], gap, hov)
#             if getZeroFlag(temp, img):
#                 ptss.append(temp.copy())
#             elif not changeFlag:
#                 gap = (-1) * gap
#                 temp = getMove(pts, gap, hov)
#                 changeFlag = 1
#                 ptss.append(temp.copy())
#             elif changeFlag:
#                 break
#         for i in ptss:
#             i = i.reshape(-1, 1, 2)
#             img = cv.polylines(img, [i], True, getColor())
#     return img


def get1LR(ptsx, img, hgap, vgap):
    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], hgap, 0)
    img = getAllTarget(ptsx, img, vgap, 1)

    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], (-1) * hgap * 2, 0)
    img = getAllTarget(ptsx, img, vgap, 1)

    return img


def get1UD(ptsx, img, hgap, vgap):
    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], vgap, 1)
    img = getAllTarget(ptsx, img, hgap, 0)

    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], (-1) * vgap * 2, 1)
    img = getAllTarget(ptsx, img, hgap, 0)

    return img


def get1Target(img, tl, best):
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

    hgap = 501
    vgap = 167
    ptsx = np.array(ptsx)
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, vgap, 1)

    for i in range(len(ptsx)):
        ptsx[i] = getMove(ptsx[i], hgap * 2, 0)
        ptsx[i] = getMove(ptsx[i], -5, 1)
    img = getAllTarget(ptsx, img, vgap, 1)

    ptsx = ptss.copy()
    ptsx = getUDMirror(ptsx, tl, 12)
    img = get1LR(ptsx, img, hgap, vgap)

    return img


def get2Target(img, tl, best):
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
    vgap = 253
    ptss = ptsx.copy()
    img = getAllTarget(ptsx, img, hgap, 0)
    ptsx = ptss.copy()
    ptsx = getLRMirror(ptsx, tl, 32)
    # img = getAllTarget(ptsx, img, 84, 0)
    img = get1UD(ptsx, img, hgap, vgap)
    ptsx = ptss.copy()
    img = get1UD(ptsx, img, hgap, vgap * 2)

    return img


def getCoordinate(img):
    oimg = img.copy()

    m = getModel()
    img = gaussianThreshold(img)

    print('进行匹配：')

    max_val, max_loc, flag, best, img = getWhich(img, m)
    oimg = cv.resize(oimg, (img.shape[1], img.shape[0]), cv.INTER_CUBIC)

    if flag == -1:
        print('Error')
        cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
        cv.imshow("match", oimg)
        return 0

    th, tw = m[flag][0].shape[:2]
    tl = max_loc

    br = (tl[0] + tw, tl[1] + th)
    cv.rectangle(oimg, tl, br, (0, 0, 255), 1)

    if not flag:
        oimg = get1Target(oimg, tl, best)
    else:
        oimg = get2Target(oimg, tl, best)

    cv.namedWindow("match", cv.WINDOW_AUTOSIZE)
    cv.imshow("match", oimg)
    return 1


def listdir(path):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


# img = cv.imread('./picture/q2/3300_TA881843BQ_TAAOL7C0_1_-191.126_-264.548__S_20180822_063225.jpg')
# start = datetime.datetime.now()
# # getCoordinate(img)
# end = datetime.datetime.now()
for i in listdir('./testp/tp'):
    getCoordinate(cv.imread(i))
    cv.waitKey(0)
    cv.destroyAllWindows()


# print('本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))

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
