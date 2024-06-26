# 车牌识别系统

## 项目描述

本项目是物联网技术与应用课程的作业，旨在实现一个车牌识别系统。车牌识别系统包括两个主要部分：车牌提取和车牌字符分割。

## 文件说明

- `lab1.py`: 包含车牌提取代码，从复杂背景中提取车牌区域。
- `lab2.py`: 包含车牌字符分割代码，从提取的车牌图像中分割出各个字符。

## 功能

1. 车牌提取：
   - 使用高斯滤波进行图像平滑处理。
   - 将图像转换为灰度图。
   - 使用Sobel算子提取图像边缘。
   - 对图像进行二值化处理。
   - 使用形态学操作去除噪声并连接字符笔画。
   - 使用轮廓检测技术检测车牌区域。
   - 筛选出符合高宽比的车牌区域。

2. 车牌字符分割：
   - 对提取的车牌图像进行去噪处理。
   - 将图像转换为灰度图。
   - 对图像进行二值化处理。
   - 使用膨胀操作连接字符笔画。
   - 使用轮廓检测技术检测字符轮廓。
   - 筛选出符合字符高宽比的轮廓。

## 环境需求

- Python 3.6+
- OpenCV库

## 安装指南

首先，确保已安装Python和pip。然后安装OpenCV库：

```bash
pip install opencv-python
```

## 运行项目

### 车牌提取

克隆仓库到本地，切换到项目目录，运行以下命令：

```bash
python lab1.py
```

### 车牌字符分割

克隆仓库到本地，切换到项目目录，运行以下命令：

```
python lab2.py
```
