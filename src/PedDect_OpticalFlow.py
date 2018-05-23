# -*- coding: utf-8 -*-

## 基于光流算法的行人轨迹检测
import numpy as np
import cv2 as cv
cap = cv.VideoCapture('.\\movie\\movie1.mp4')

# ShiTomasi角点检测参数设置
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.05,
                       minDistance = 5,
                       blockSize = 5 )

# lucas-kanade光流法参数设置
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 选取第一帧并寻找角点
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

mask = np.zeros_like(old_frame)
origin = np.zeros_like(old_frame)
color = np.random.randint(0, 255, (500, 3))
while(1):
    ret, frame = cap.read()
    if(ret == False):
        break
    origin = frame.copy()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择运动的角点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        mask = cv.line(mask, (a,b), (c,d), [0, 0, 255], 2)
        frame = cv.circle(frame, (a,b), 2, [0, 0, 255], -1)

    img = cv.add(frame, mask)

    # 图片显示
    cv.imshow('img', img) # 在原图上实时显示轨迹
    cv.imshow('mask', mask) # 轨迹

    # 检测ESC键退出
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # 更新old_gray
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# 图片保存
cv.imwrite(".\\pic\\mask.jpg", mask)
cv.imwrite(".\\pic\\img.jpg", img)
cv.imwrite(".\\pic\\origin.jpg", origin)

# 关闭窗口并释放资源
cv.destroyAllWindows()
cap.release()
