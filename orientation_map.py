import cv2

# 读取图像
image = cv2.imread('00018_33.png', cv2.IMREAD_GRAYSCALE)

# 计算x和y方向的梯度
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算方向图
orientation_map = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)

# 将方向图转换为8位
orientation_8u = cv2.convertScaleAbs(orientation_map)

# 彩色显示方向图
colored_orientation_map = cv2.applyColorMap(orientation_8u, cv2.COLORMAP_JET)

# 显示彩色方向图
cv2.imshow('Colored Orientation Map', colored_orientation_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
