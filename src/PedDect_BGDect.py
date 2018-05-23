# -*- coding: utf-8 -*-

## 基于目标跟踪-背景分割的行人轨迹检测
import numpy as np
import cv2 as cv

# 图片读取
cap = cv.VideoCapture('C:\\Users\\Administrator\\Desktop\\PedDect\\PedDect_Py\\PedDect_Py\\movie\\承德避暑山庄德汇门入口2.mp4')
BackGroundModel = cv.createBackgroundSubtractorMOG2() # 背景分割器MOG2
ret, frame = cap.read()
track = np.zeros_like(frame)
while(1):
    ret, frame = cap.read()
    if(ret == False):
        break
    originFrame = frame.copy() # 原始图片

    # 背景分割器，计算前景掩码
    fgmask = BackGroundModel.apply(frame) 
    
    # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值
    foreGround = cv.threshold(fgmask.copy(), 254, 255, cv.THRESH_BINARY)[1]
    
    # 中值滤波，滤除噪点
    foreGround = cv.medianBlur(foreGround, 5)
    
    # 计算目标轮廓
    image, contours, hierarchy = cv.findContours(foreGround, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # 选取轮廓面积大于 30 的，绘制轨迹线条
    for c in contours:
        if 150 > cv.contourArea(c) > 30:
            (x, y, w, h) = cv.boundingRect(c)
            # cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            track = cv.line(track, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 4)

    PedDect_img = cv.add(frame, track)
    
    # 图片显示
    cv.imshow('track', track) # 检测到的轨迹
    cv.imshow('foreGround', foreGround) # 提取的前景图像
    cv.imshow('PedDect_img', PedDect_img) # 在原始图像中实时显示轨迹

    # ESC键退出
    key = cv.waitKey(30) & 0xff
    if key == 27:
        break

# 图片保存
cv.imwrite(".\\pic\\track.jpg", track) 
cv.imwrite(".\\pic\\foreGround.jpg", foreGround)
cv.imwrite(".\\pic\\originFrame.jpg", originFrame)
cv.imwrite(".\\pic\\PedDect_img.jpg", PedDect_img)

cv.destroyAllWindows()
cap.release()
