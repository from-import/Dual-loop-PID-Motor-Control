import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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

# PID控制器参数调整
Kp_s = 20
Ki_s = 10
Kd_s = 20

Kp_i = 1.5
Ki_i = 0.8
Kd_i = 0.3

# 仿真时间设置
t = np.linspace(0, 5, 1000)  # 5秒，1000步
dt = t[1] - t[0]

# 参考信号
desired_speed = 100 # 期望速度 (rad/s)
desired_current = 0.5 * K_t * desired_speed / K_e  # 期望电流 (A)

# 初始状态变量 [omega, i] (角速度, 电流)
initial_conditions = [0,0]

# 系统方程 (ODEs)
def motor_system(state, t, Kp_s, Ki_s, Kd_s, Kp_i, Ki_i, Kd_i, desired_speed, desired_current):
    omega, i = state
    speed_error = desired_speed - omega
    current_error = desired_current - i

    # 速度环PID
    speed_integral = Ki_s * speed_error * dt
    speed_derivative = Kd_s * (speed_error - speed_error) / dt
    speed_control = Kp_s * speed_error + speed_integral + speed_derivative

    # 电流环PID
    current_integral = Ki_i * current_error * dt
    current_derivative = Kd_i * (current_error - current_error) / dt
    current_control = Kp_i * current_error + current_integral + current_derivative

    # 电机动力学
    domega_dt = (K_t * i - b * omega) / J
    di_dt = (speed_control - K_e * omega - R * i) / L

    return [domega_dt, di_dt]

# 解ODEs
result = odeint(motor_system, initial_conditions, t, args=(Kp_s, Ki_s, Kd_s, Kp_i, Ki_i, Kd_i, desired_speed, desired_current))

# 绘图结果
plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(t, result[:, 0], label='角速度 (rad/s)')
plt.plot(t, [desired_speed] * len(t), 'r--', label='期望速度 (rad/s)')
plt.title('速度响应')
plt.xlabel('时间 (s)')
plt.ylabel('速度 (rad/s)')
plt.legend()

plt.subplot(212)
plt.plot(t, result[:, 1], label='电机电流 (A)')
plt.plot(t, [desired_current] * len(t), 'r--', label='期望电流 (A)')
plt.title('电流响应')
plt.xlabel('时间 (s)')
plt.ylabel('电流 (A)')
plt.legend()

plt.tight_layout()
plt.show()
