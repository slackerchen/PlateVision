# 车牌提取系统

## 项目描述
这个项目是物联网技术与应用课程的一部分，旨在开发一个车牌提取系统。使用Python和OpenCV库，实现从车辆图片中自动检测并提取车牌信息。

## 功能
- 使用高斯滤波器进行图像平滑处理。
- 转换图像到灰度以减少处理时间。
- 应用Sobel算子提取图像边缘。
- 通过形态学操作清除噪声并强化车牌区域的特征。
- 使用轮廓检测技术来定位车牌。

## 环境需求
- Python 3.6+
- OpenCV库

## 安装指南
首先，确保已安装Python和pip。然后安装OpenCV库：
```bash
pip install opencv-python
```

## 运行项目

克隆仓库到本地，切换到项目目录，运行以下命令：

```bash
python license_plate_extractor.py
```

