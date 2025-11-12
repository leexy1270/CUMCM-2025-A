import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from test1 import test1, direction
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正常显示

# 初始坐标
True_obj = np.array([0, 200, 0])
F = np.array([0, 0, 0])
M1 = np.array([20000, 0, 2000])
M2 = np.array([19000, 600, 2100])
M3 = np.array([18000, -600, 700])
FY1 = np.array([17800, 0, 1800])

# 目标函数 - 计算有效遮蔽时间
def objective_function(x):
    # 解包参数: 无人机速度, 投放时间, 起爆时间, 无人机航向角
    v_drone, A, B, theta = x

    # 无人机飞行方向 (水平于xy平面)
    drone_direction = np.array([np.cos(theta), np.sin(theta), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction = drone_direction

    # 导弹飞行方向
    missile_direction = direction(M1, F)
    v_missile = 300  # 导弹速度300m/s

    # 时间参数
    T = np.linspace(0, 20, 2000)
    t = A + B + T

    # 计算结果
    result = []
    for i in t:
        # 计算烟幕弹位置 (基于原始代码的计算方式)
        n_jkt = np.array(FY1) + (A + B) * v_drone * bomb_direction - 0.5 * 9.8 * (B) ** 2 * np.array([0, 0, 1]) - 3 * (
                    i - A - B) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction * v_missile * i

        # 检验
        result.append(test1(M1_t, n_jkt))

    result = np.array(result)
    total_time = np.sum(result) * (20 / 2000)

    # 我们希望最大化总时间，但优化算法默认最小化，所以取负
    return -total_time


# 定义参数边界
# v_drone, A, B, theta
bounds = [(70, 140), (0.1, 5), (0.1, 5), (0, 2 * np.pi)]

# 使用双模拟退火算法进行优化
result = dual_annealing(objective_function, bounds, maxiter=100, seed=42)
xopt = result.x
fopt = result.fun

print("模拟退火优化结果:")
print(f"最佳参数: 无人机速度={xopt[0]} m/s, 投放时间={xopt[1]} s, 起爆时间={xopt[2]} s, 航向角={xopt[3]} rad")
print(f"最大有效遮蔽时间: {-fopt} 秒")

# 使用最佳参数计算并可视化结果
v_drone_opt, A_opt, B_opt, theta_opt = xopt
drone_direction_opt = np.array([np.cos(theta_opt), np.sin(theta_opt), 0])
bomb_direction_opt = drone_direction_opt  # 投弹方向与无人机方向一致

# 导弹飞行方向
missile_direction = direction(M1, F)
v_missile = 300

# 时间参数
T = np.linspace(0, 20, 2000)
t = A_opt + B_opt + T
result_opt = []

for i in t:
    n_jkt = np.array(FY1) + (A_opt + B_opt) * v_drone_opt * bomb_direction_opt - 0.5 * 9.8 * (B_opt) ** 2 * np.array(
        [0, 0, 1]) - 3 * (i - A_opt - B_opt) * np.array([0, 0, 1])
    M1_t = np.array(M1) + missile_direction * v_missile * i
    result_opt.append(test1(M1_t, n_jkt))

result_opt = np.array(result_opt)

plt.figure(figsize=(10, 6))
plt.plot(T, result_opt)
plt.xlabel('时间 (s)')
plt.ylabel('是否在目标区域')
plt.title('优化后炸弹在目标区域的时间')
plt.grid(True)
plt.show()

print(f"优化后在目标区域总时间: {np.sum(result_opt) * (20 / 2000)} 秒")

