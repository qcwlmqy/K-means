import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_SIFT(img, keypoints):
    # 将keypoints绘制到原图中
    img_sift = np.copy(img)
    cv2.drawKeypoints(img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 显示绘制有特征点的图像
    plt.figure(12, figsize=(15, 30))
    plt.subplot(121)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Raw Img')

    plt.subplot(122)
    img_sift_rgb = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    plt.imshow(img_sift_rgb)
    plt.title('Img with SIFT features')
    plt.show()


# 提取 SIFT 特征
def extract_SIFT(root):
    features = []
    # sift 提取器
    sift = cv2.xfeatures2d.SIFT_create()

    for home, dirs, files in os.walk(root):
        for img_type in dirs:
            img_dir = os.path.join(root, img_type)
            for img in os.listdir(img_dir):
                if img.startswith('H1'):
                    continue
                full_file = os.path.join(img_dir, img)
                image = cv2.imread(full_file, cv2.IMREAD_COLOR)
                # 提取图像的 SIFT
                keypoint, feature = sift.detectAndCompute(image, None)
                features.append(feature)
    return np.array(features)


def K_means(features):
    type_num = 8
    # Kmeans
    compactness, labels, centers = cv2.kmeans(features,
                                              type_num,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1),
                                              20,
                                              cv2.KMEANS_RANDOM_CENTERS)
    return centers


if __name__ == '__main__':
    features = extract_SIFT(r'D:\Learn\Vision Introduction\dataset')
    centers = K_means(features)
    print('end')

    # print("Hello")
    # sift = cv2.xfeatures2d.SIFT_create()  # 构建SIFT特征点检测器对象
    # keypoints = sift.detect('GRAY', None)  # 用SIFT特征点检测器对象检测灰度图中的特征点
