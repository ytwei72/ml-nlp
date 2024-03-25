from test_whl import test_whl_fn1

# test_whl_fn1()
from common.np import *

# Python version
import sys
print('Python: {}'.format(sys.version))

import cupy as cp
cp.cuda.Device(device_id)

# 创建 CuPy 数组
arr = cp.array([1, 2, 3, 4, 5])

# 逐个获取元素值
for i in range(arr.size):
    value = arr[i]
    # value = arr.get(i)
    print(value)

# 创建一个3维数组，包含两个2x3的矩阵
# arr = np.array([[[1, 2, 3],
#                  [4, 5, 6]],
#                 [[7, 8, 9],
#                  [10, 11, 12]]])
# arr = np.array([[[11, 2, 3],
#                  [15, 5, 6]],
#                 [[7, 11, 9],
#                  [10, 15, 3]]])
# arr = np.array([[[11, 2],
#                  [15, 5]],
#                 [[7, 11],
#                  [10, 15]]])
# arr = np.array([[1, 2, 3],
#                  [4, 5, 6]])

# 使用argmax(axis=2)获取每个矩阵中最大值所在的列索引
# result = np.argmax(arr, axis=2)

# print(result)

# import os
# print(os.environ)

# arr = np.array([1, 2, 3, 4, 5])
# mask = np.array([True, False, True, False, True])
#
# # 使用布尔索引设置值
# arr[~mask] = 0
# # arr *= mask
#
# print(arr)

# for i in range(20):
#     arr = np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
#     print(arr)

# a = np.array([[1, 2], [3, 4]])
# b = np.array([True, False])
# a *= b
# print(a)
#
# from common.np import *
#
#
# a = np.zeros((3, 3), dtype=np.int64)
# b = list([1,2,3])
# a[0] = b
