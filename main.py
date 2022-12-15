import os
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sum_type = 3


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
    features = np.float32([]).reshape(0, 128)
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
                # show_SIFT(image, keypoint)
                features = np.append(features, feature, axis=0)
    return features


def extract_img_SIFT(root):
    sift = cv2.xfeatures2d.SIFT_create()
    image = cv2.imread(root, cv2.IMREAD_COLOR)
    keypoint, feature = sift.detectAndCompute(image, None)
    return feature


def K_means(features, randomState=None):
    # Kmeans
    kMeans = KMeans(n_clusters=sum_type, random_state=randomState)
    kMeans.fit(features)
    return kMeans.labels_, kMeans.cluster_centers_


def calcFeatVec(features, centers, type):
    featVec = np.zeros((1, sum_type))
    for i in range(0, features.shape[0]):
        fi = features[i]
        diffMat = np.tile(fi, (sum_type, 1)) - centers
        # sqSum = (diffMat ** 2).sum(axis=1)
        if type == 'l2':
            diffMat = diffMat ** 2
        sqSum = diffMat.sum(axis=1)
        dist = sqSum
        sortedIndices = dist.argsort()
        idx = sortedIndices[0]
        featVec[0][idx] += 1
    return featVec


if __name__ == '__main__':
    features = extract_SIFT(r'D:\Learn\Vision Introduction\dataset')
    _, centers = K_means(features)

    root = r'test'
    s = {}
    for img in os.listdir(root):
        img_type = img[:-5]
        full_file = os.path.join(root, img)
        feature = extract_img_SIFT(full_file)
        featVec = calcFeatVec(feature, centers, 'l2')
        s[numpy.argmax(featVec)] = img_type
        image = cv2.imread(full_file, cv2.IMREAD_COLOR)
    print(s)

    for img in os.listdir(root):
        img_type = img[:-5]
        full_file = os.path.join(root, img)
        feature = extract_img_SIFT(full_file)
        featVec = calcFeatVec(feature, centers, 'l1')
        print("label is {0}, pre_label is {1}".format(img_type, s[numpy.argmax(featVec)]))
        image = cv2.imread(full_file, cv2.IMREAD_COLOR)

    for img in os.listdir(root):
        img_type = img[:-5]
        full_file = os.path.join(root, img)
        feature = extract_img_SIFT(full_file)
        featVec = calcFeatVec(feature, centers, 'l2')
        print("label is {0}, pre_label is {1}".format(img_type, s[numpy.argmax(featVec)]))

        image = cv2.imread(full_file, cv2.IMREAD_COLOR)

    print('end')
