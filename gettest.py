import os,shutil
import random
from getit2 import getCoordinate, getJPG
import datetime
import cv2 as cv

def getPathPic(srcpath, path):
    # 返回一个文件夹下所有jpg文件名，并随机抽取30%作为test
    list_name = []
    for file in os.listdir(srcpath):
        file_path = os.path.join(srcpath, file)
        if not os.path.isdir(file_path) and file[-3:].lower() == 'jpg' and random.randint(1, 10) < 4:
            shutil.move(file_path, os.path.join(path, file))
    print(srcpath + ' Done')

def getPic(path):
    # MMG3350Defect+Code图片
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TGGS0/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TPDEB/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TPDPD/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TPDPL/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TPDPS/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u7247/TPFIP/', path)

    # MMG3350Defect+Code图片02
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TPOTS/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TPWR0/', path)
    # getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TSDFS/', path) # 模糊
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TSFAS/', path)
    # getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TSFIX/', path) # 半截
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724702/TSILR/', path)

    # MMG3350Defect+Code图片03
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTFBA/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTFBS/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTLSA/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTMUD/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTP3S/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724703/TTSAD/', path)

    # MMG3350Defect+Code图片04
    # getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724704/TTSPS/', path) # 有图片分辨率奇怪
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724704/TTUNW/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724704/TTWRG/', path)
    getPathPic('./3350MMG/MMG3350Defect+Code\u56fe\u724704/TTWRS/', path)

def clearDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)

def main():
    samppath = './sampTestPic/'
    clearDir(samppath)
    outpath = './outTestPic/'
    clearDir(outpath)

    getPic(samppath)

    all_sum = 0
    error_sum = 0
    all_start = datetime.datetime.now()
    for i in getJPG(samppath):
        all_sum += 1
        start = datetime.datetime.now()
        out = getCoordinate(cv.imread(i), 1)
        end = datetime.datetime.now()
        if not out[0]:
            error_sum += 1
        print('    本次匹配费时%fs:' % (((end - start).microseconds) / 1e6))
        cv.imwrite(i.replace(samppath[1:], outpath[1:]) ,out[1])
    all_end = datetime.datetime.now()
    print('一共处理了%d张图片，失败%d张，处理率为%f，共耗时%fs' % (all_sum, error_sum, (1-(error_sum/all_sum)), ((all_end - all_start).seconds)))

if __name__ == '__main__':
    main()
 