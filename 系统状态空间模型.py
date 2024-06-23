import numpy as np


# 电机和控制系统参数
K_t = 0.1  # 转矩常数 (Nm/A)
K_e = 0.1  # 反电动势常数 (V/rad/s)
R = 0.1   # 电阻 (欧姆)
L = 1   # 电感 (H)
J = 0.01   # 转动惯量 (kg.m^2)
b = 0.05   # 阻尼比 (Nms)

# 状态空间矩阵
A = np.array([[-b/J, K_t/J],
              [-K_e/L, -R/L]])
B = np.array([[0],
              [1/L]])
C = np.eye(2)
D = np.zeros((2, 1))

# 打印状态空间矩阵
print("状态矩阵 A:\n", A)
print("输入矩阵 B:\n", B)
print("输出矩阵 C:\n", C)
print("传递矩阵 D:\n", D)
