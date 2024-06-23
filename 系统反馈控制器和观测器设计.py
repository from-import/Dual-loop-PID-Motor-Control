import numpy as np
from scipy.signal import place_poles, lti, lsim
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

# 期望闭环极点（可以根据需要调整）
desired_poles = np.array([-10, -20])

# 计算状态反馈增益矩阵K
place_obj = place_poles(A, B, desired_poles)
K = place_obj.gain_matrix
print("状态反馈增益矩阵K:\n", K)

# 设计观测器
# 期望观测器极点（应比闭环极点快）
desired_observer_poles = np.array([-30, -40])

# 计算观测器增益矩阵L
place_obj_observer = place_poles(A.T, C.T, desired_observer_poles)
L = place_obj_observer.gain_matrix.T
print("观测器增益矩阵L:\n", L)

# 新的状态空间矩阵（带观测器和状态反馈的系统）
A_cl = np.block([[A - B @ K, B @ K],
                 [np.zeros_like(A), A - L @ C]])
B_cl = np.block([[B],
                 [np.zeros_like(B)]])
C_cl = np.block([C, np.zeros_like(C)])
D_cl = D

# 创建新的状态空间系统
system_cl = lti(A_cl, B_cl, C_cl, D_cl)

# 仿真时间设置
t = np.linspace(0, 5, 1000)  # 5秒，1000步
u = np.zeros_like(t)  # 输入为零（测试观测器性能）

# 初始状态
x0 = [30, 100, 0, 0]  # 假设初始状态偏离实际状态

# 进行仿真
t, y, x = lsim(system_cl, u, t, X0=x0)

# 绘制角速度和电流的响应
plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.plot(t, y[:, 0], label='角速度 (rad/s)')
plt.title('角速度响应（带观测器和状态反馈）')
plt.xlabel('时间 (s)')
plt.ylabel('角速度 (rad/s)')
plt.legend()

plt.subplot(212)
plt.plot(t, y[:, 1], label='电流 (A)')
plt.title('电流响应（带观测器和状态反馈）')
plt.xlabel('时间 (s)')
plt.ylabel('电流 (A)')
plt.legend()

plt.tight_layout()
plt.show()
