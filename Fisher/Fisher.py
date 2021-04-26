from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 设置随机数种子，保证每次产生相同的数据。
X, y = make_multilabel_classification(n_samples=2000, n_features=2, n_labels=1, n_classes=1, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# 1、根据类别分为两类
index1 = np.array([index for (index, value) in enumerate(y_train) if value == 0])  # 获取类别1的indexs
index2 = np.array([index for (index, value) in enumerate(y_train) if value == 1])  # 获取类别2的indexs

c_1 = X_train[index1]   # 类别1的所有数据(x1, x2) in X_1
c_2 = X_train[index2]   # 类别2的所有数据(x1, x2) in X_2

# 2、Fisher算法实现
def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples:
    :return:
    """
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t*t.reshape(2, 1)   
    return cov_m, u1
    
def fisher(c_1, c_2):
    """
    fisher算法实现(参考上面的推导公式进行理解)
    :param c_1:
    :param c_2:
    :return:
    """
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2          # 总类内离散度矩阵。
    u, s, v = np.linalg.svd(s_w) # 下面的参考公式（4-10）
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)

# 3、判断类别
def judge(sample, w, c_1, c_2):
    """
    返回值：ture 属于1；false 属于2
    :param sample:
    :param w:
    :param c_1:
    :param c_2:
    :return:
    """
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1) # 参考公式(2-8)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)  # 新样本进来判断
    return abs(pos - center_1) < abs(pos - center_2)

w = fisher(c_1, c_2)             # 调用函数，得到参数w
out = []
for i in range(len(y_test)):
    out.append (judge(X_test[i], w, c_1, c_2)) # 判断所属的类别。
print(out)

index1_t = np.array([index for (index, value) in enumerate(out) if value == False])  # 获取类别1的indexs
index2_t = np.array([index for (index, value) in enumerate(out) if value == True])
c_1t = X_test[index1_t]   
c_2t = X_test[index2_t]

# 4、绘图功能
plt.figure(1)
plt.scatter(c_1[:, 0], c_1[:, 1], c='red')
plt.scatter(c_2[:, 0], c_2[:, 1], c='blue')
# plt.scatter(c_1t[:, 0], c_1t[:, 1], c='yellow')
# plt.scatter(c_2t[:, 0], c_2t[:, 1], c='green')
'''line_x = np.arange(-min(np.min(c_1[:, 0]), np.min(c_2[:, 0])),
                   max(np.max(c_1[:, 0]), np.max(c_2[:, 0])),
                   step=1)'''
line_x = np.arange(-10, 40, step=1)
line_y = (w[1]*line_x) / w[0]
plt.plot(line_x, line_y, linewidth=3.0,  label = 'fisher boundary line ')
plt.legend(loc='upper right')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Training result')
plt.show()

plt.figure(2)
plt.scatter(c_1[:, 0], c_1[:, 1], c='red')
plt.scatter(c_2[:, 0], c_2[:, 1], c='blue')
plt.scatter(c_1t[:, 0], c_1t[:, 1], c='yellow')
plt.scatter(c_2t[:, 0], c_2t[:, 1], c='green')
'''line_x = np.arange(-min(np.min(c_1[:, 0]), np.min(c_2[:, 0])),
                   max(np.max(c_1[:, 0]), np.max(c_2[:, 0])),
                   step=1)'''
line_x = np.arange(-10, 40, step=1)
line_y = (w[1]*line_x) / w[0]
plt.plot(line_x, line_y, linewidth=3.0,  label = 'fisher boundary line ')
plt.legend(loc='upper right')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Testing result')
plt.show()
