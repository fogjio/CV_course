import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_centroies (df, k, centroids):
    for c in range(1,k):
        # 求出所有点到现有的所有中心点的距离。作为求概率的数据基础
        distance_sum, data_temp = distance_AllCentroies(df, centroids)
        # 求概率  新建prob列存放概率
        data_temp['prob'] = data_temp['allcentroid_sum'] / distance_sum
        # 建立prob_add存放累加求和概率，用于轮盘赌选择中心点
        data_temp.loc[0, 'prob_add'] = data_temp.loc[0, 'prob']
        for i in range(1, len(data_temp['x'])):
            data_temp.loc[i, ['prob_add']] = (data_temp.loc[i, 'prob'] + data_temp.loc[i - 1, 'prob_add'])
        # 中心点选择过程
        a = np.random.rand()
        for idxs in range(len(data_temp)):
            if a < data_temp.loc[idxs,'prob_add']:
                index_c = idxs
                break
        # 添加中心点
        centroids[c] = [df.loc[idxs, 'x'], df.loc[idxs, 'y']]
    return centroids

def distance_AllCentroies (df, centroids):
    # 深度拷贝用来计算，不破坏初始数据
    data_temp = df.copy(deep=True)
    # 对每个中心点求和
    for i in centroids.keys():
        data_temp['distance_from_{}'.format(i)] = (data_temp['x'] - centroids[i][0]) ** 2 + (data_temp['y'] - centroids[i][1]) ** 2
    # 找出已经计算数据的中心点的列名
    Allcentroes_index_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    data_temp['allcentroid_sum'] = data_temp.loc[:, Allcentroes_index_id].apply(lambda x: x.sum(), axis=1)
    aa = data_temp.apply(lambda x: x.sum())
    # 创建出一列存放距离和
    distance_sum = aa['allcentroid_sum']
    return distance_sum, data_temp



def assignment (df, centroids, colmap):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt((df['x'] - centroids[i][0]) ** 2 + (df['y'] - centroids[i][1]) ** 2)
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)    #每一行的最小值  axis = 1
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df


def updata(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def main():
    #直接返回一个二维矩阵
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3
    # 建立一个字典类型存储中心点，这里随机找一个初始中心点
    init_cindex = np.random.randint(df['x'].size)
    centroids = {0: [df.loc[init_cindex, 'x'], df.loc[init_cindex, 'y']]}
    get_centroies(df, k, centroids)

    # 颜色字典
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = updata(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha = 0.5, edgecolors = 'k')
        for i in centroids.keys():
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.xlim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)
        if closest_centroids.equals(df['closest']):
            break



if __name__ == '__main__':
    main()