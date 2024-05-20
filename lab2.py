import cv2
import numpy as np


def segment_license_plate(plate_image_path):
    # 读取车牌图像
    plate_image = cv2.imread(plate_image_path)

    # 步骤1: 高斯滤波去噪
    image_blur = cv2.GaussianBlur(plate_image, (5, 5), 0)

    # 步骤2: 色彩空间转换 BGR to GRAY
    gray_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    # 步骤3: 阈值处理，转换为二值图像
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 步骤4: 膨胀操作连接字符笔画
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)

    # 步骤5: 查找图像内的所有轮廓
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 步骤6: 使用boundingRect对每个轮廓用矩形包围框包围
    char_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        # 假设字符的高宽比在0.2到0.9之间，面积大于一定值
        if 0.2 < aspect_ratio < 0.9 and h > 15 and area > 100:
            char_candidates.append((x, y, w, h))

    # 步骤7: 遍历包围框，筛选字符轮廓
    plate_chars = []
    for (x, y, w, h) in sorted(char_candidates, key=lambda item: item[0]):
        char_image = binary_image[y:y + h, x:x + w]
        plate_chars.append(char_image)
        cv2.rectangle(plate_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 步骤8: 输出各个字符
    cv2.imshow('Segmented Plate', plate_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i, char_img in enumerate(plate_chars):
        cv2.imshow(f'Char {i + 1}', char_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plate_chars


# 使用函数
plate_image_path = 'Plate/p1.jpg'
segmented_chars = segment_license_plate(plate_image_path)
