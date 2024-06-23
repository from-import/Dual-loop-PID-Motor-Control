import numpy as np
from scipy.signal import lti, ss2tf, bode, step, TransferFunction
import matplotlib.pyplot as plt

# 设置中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 创建状态空间系统
system = lti(A, B, C, D)

# 计算系统的特征值（闭环极点）
eigvals, eigvecs = np.linalg.eig(A)
print("系统特征值:\n", eigvals)

# 时域响应（单位阶跃响应）
t, y = step(system)

# 绘制角速度和电流的阶跃响应
plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.plot(t, y[:, 0], label='角速度 (rad/s)')
plt.title('角速度阶跃响应')
plt.xlabel('时间 (s)')
plt.ylabel('角速度 (rad/s)')
plt.legend()

plt.subplot(212)
plt.plot(t, y[:, 1], label='电流 (A)')
plt.title('电流阶跃响应')
plt.xlabel('时间 (s)')
plt.ylabel('电流 (A)')
plt.legend()

plt.tight_layout()
plt.show()

# 将状态空间模型转换为传递函数
num_omega, den_omega = ss2tf(A, B, C[0], D[0])
num_current, den_current = ss2tf(A, B, C[1], D[1])

# 创建传递函数系统
tf_omega = TransferFunction(num_omega, den_omega)
tf_current = TransferFunction(num_current, den_current)

# 频域响应（Bode图）
w, mag_omega, phase_omega = bode(tf_omega)
_, mag_current, phase_current = bode(tf_current)

plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.semilogx(w, mag_omega, label='角速度 (rad/s)')
plt.title('Bode 图 - 幅值')
plt.xlabel('频率 (rad/s)')
plt.ylabel('幅值 (dB)')
plt.legend()

plt.subplot(212)
plt.semilogx(w, phase_omega, label='角速度 (rad/s)')
plt.title('Bode 图 - 相位')
plt.xlabel('频率 (rad/s)')
plt.ylabel('相位 (度)')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.semilogx(w, mag_current, label='电流 (A)')
plt.title('Bode 图 - 幅值')
plt.xlabel('频率 (rad/s)')
plt.ylabel('幅值 (dB)')
plt.legend()

plt.subplot(212)
plt.semilogx(w, phase_current, label='电流 (A)')
plt.title('Bode 图 - 相位')
plt.xlabel('频率 (rad/s)')
plt.ylabel('相位 (度)')
plt.legend()

plt.tight_layout()
plt.show()