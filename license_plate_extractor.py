import cv2
import numpy as np


def extract_license_plate(image_path):
    # 步骤1: 读取原始图像
    image = cv2.imread(image_path)
    if image is None:
        print("图像文件未找到")
        return None

    # 步骤2: 高斯滤波
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Gaussian Blur', image_blur)
    cv2.waitKey(0)

    # 步骤3: 色彩空间转换 BGR to GRAY
    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray_image)
    cv2.waitKey(0)

    # 步骤4: 使用Sobel算子提取边缘
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    edge_image = np.uint8(abs_sobelx)
    cv2.imshow('Edge Image', edge_image)
    cv2.waitKey(0)

    # 步骤5: 二值化处理
    _, binary_image = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)

    # 步骤6: 形态学操作 - 先闭运算后开运算
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Morphological Transformations', opening)
    cv2.waitKey(0)

    # 步骤7: 中值滤波
    denoised_image = cv2.medianBlur(opening, 5)
    cv2.imshow('Denoised Image', denoised_image)
    cv2.waitKey(0)

    # 步骤8: 查找轮廓
    contours, _ = cv2.findContours(denoised_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)

    # 步骤9: 筛选车牌轮廓
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 3 < aspect_ratio < 10:  # 根据车牌的标准尺寸筛选
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            plate = image[y:y + h, x:x + w]
            cv2.imshow('License Plate', plate)
            cv2.waitKey(0)
            return plate

    cv2.destroyAllWindows()
    return None


# 使用函数
plate_image = extract_license_plate('Plate/p1.jpg')
if plate_image is not None:
    cv2.imshow('Extracted Plate', plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
