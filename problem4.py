import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from test1 import test1, direction

# 移除可能导致问题的字体设置
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei"]

# 初始坐标
True_obj = np.array([0, 200, 0])
F = np.array([0, 0, 0])
M1 = np.array([20000, 0, 2000])
M2 = np.array([19000, 600, 2100])
M3 = np.array([18000, -600, 700])
FY1 = np.array([17800, 0, 1800])
FY2 = np.array([12000, 1400, 1400])
FY3 = np.array([6000, -3000, 700])


# 目标函数 - 计算有效遮蔽时间
def objective_function(x):
    # 解包参数: 无人机速度1, 无人机航向角1,投放时间1,起爆时间1, 无人机速度2, 无人机航向角2,投放时间2,起爆时间2,无人机速度3, 无人机航向角3,投放时间3,起爆时间3
    v_drone1, theta1, A1, B1,v_drone2, theta2, A2, B2,v_drone3, theta3, A3, B3 = x

    # 导弹飞行方向
    missile_direction = direction(M1, F)
    v_missile = 300  # 导弹速度300m/s

    # 无人机1飞行方向 (水平于xy平面)
    drone_direction1 = np.array([np.cos(theta1), np.sin(theta1), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction1 = drone_direction1

    # 无人机2飞行方向 (水平于xy平面)
    drone_direction2 = np.array([np.cos(theta2), np.sin(theta2), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction2 = drone_direction2

    # 无人机3飞行方向 (水平于xy平面)
    drone_direction3 = np.array([np.cos(theta3), np.sin(theta3), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction3 = drone_direction3


    # 时间参数
    total_missile_time = np.linalg.norm(M1 - F) / v_missile
    T = np.arange(0, total_missile_time, 0.01)

    # 计算结果 - 直接使用列表，不使用 pandas Series
    timeline = np.zeros(len(T))

    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A1 + B1:
            continue

        # 计算烟幕弹位置
        n_jkt1 = np.array(FY1) + (A1 + B1) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B1) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                         t - A1 - B1) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction * v_missile * t

        # 检验
        if test1(M1_t, n_jkt1):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A2 +B2:
            continue

        # 计算烟幕弹位置
        n_jkt2 = np.array(FY2) + (A2 + B2) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B2) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - B2 - A2) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction * v_missile * t

        # 检验
        if test1(M1_t, n_jkt2):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A3 + B3:
            continue

        # 计算烟幕弹位置
        n_jkt3 = np.array(FY3) + ( A3 + B3) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B3) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - B3 - A3) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction * v_missile * t

        # 检验
        if test1(M1_t, n_jkt3):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 我们希望最大化总时间，但优化算法默认最小化，所以取负
    return -total_time


# 定义参数边界
# v_drone1, theta1, A1, B1,v_drone2, theta2, A2, B2,v_drone3, theta3, A3, B3
bounds = [(70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7),(70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7)]



# PSO算法实现
class PSO:
    def __init__(self, objective_func, bounds, num_particles=40, max_iter=260, w=0.9, c1=1.5, c2=1.5):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.dim = len(bounds)

        # 初始化粒子位置和速度
        self.X = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds],
                                   size=(num_particles, self.dim))
        self.V = np.random.uniform(-1, 1, (num_particles, self.dim))

        # 初始化个体最优位置和适应度
        self.P = self.X.copy()
        self.p_fitness = np.array([self.objective_func(x) for x in self.X])

        # 初始化全局最优位置和适应度
        self.g = self.P[np.argmin(self.p_fitness)]
        self.g_fitness = np.min(self.p_fitness)

        # 记录每次迭代的最佳适应度
        self.fitness_history = []

    def optimize(self):
        for iter in range(self.max_iter):
            # 更新惯性权重（线性递减）
            w = self.w * (1 - iter / self.max_iter)

            for i in range(self.num_particles):
                # 更新速度
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.V[i] = (w * self.V[i] +
                             self.c1 * r1 * (self.P[i] - self.X[i]) +
                             self.c2 * r2 * (self.g - self.X[i]))

                # 更新位置
                self.X[i] = self.X[i] + self.V[i]

                # 边界处理
                for j in range(self.dim):
                    if self.X[i, j] < self.bounds[j][0]:
                        self.X[i, j] = self.bounds[j][0]
                    elif self.X[i, j] > self.bounds[j][1]:
                        self.X[i, j] = self.bounds[j][1]

                # 计算新位置的适应度
                fitness = self.objective_func(self.X[i])

                # 更新个体最优
                if fitness < self.p_fitness[i]:
                    self.P[i] = self.X[i].copy()
                    self.p_fitness[i] = fitness

                # 更新全局最优
                if fitness < self.g_fitness:
                    self.g = self.X[i].copy()
                    self.g_fitness = fitness

            # 记录当前迭代的最佳适应度
            self.fitness_history.append(self.g_fitness)

            # 打印当前迭代信息
            if (iter + 1) % 10 == 0:
                print(f"Iteration {iter + 1}/{self.max_iter}, Best Fitness: {-self.g_fitness:.4f}")

        return self.g, self.g_fitness


# 使用PSO算法进行优化
print("开始PSO优化...")
pso = PSO(objective_function, bounds, num_particles=50, max_iter=250)
xopt, fopt = pso.optimize()

print("PSO优化结果:")
print(
    f"最佳参数: 无人机速度1:{xopt[0]}, 无人机航向角1:{xopt[1]},投放时间1:{xopt[2]},起爆时间1:{xopt[3]}, 无人机速度2:{xopt[4]}, 无人机航向角2:{xopt[5]},投放时间2:{xopt[6]},起爆时间2:{xopt[7]},无人机速度3:{xopt[8]}, 无人机航向角3:{xopt[9]},投放时间3:{xopt[10]},起爆时间3:{xopt[11]}")
print(f"最大有效遮蔽时间: {-fopt} 秒")

# 使用最佳参数计算并可视化结果
v_drone1_opt, theta1_opt, A1_opt, B1_opt, v_drone2_opt, theta2_opt, A2_opt, B2_opt, v_drone3_opt, theta3_opt, A3_opt, B3_opt = xopt

# 导弹飞行方向
missile_direction = direction(M1, F)
v_missile = 300  # 导弹速度300m/s

# 无人机1飞行方向 (水平于xy平面)
drone_direction1_opt = np.array([np.cos(theta1_opt), np.sin(theta1_opt), 0])

# 投弹方向与无人机飞行方向一致
bomb_direction1_opt = drone_direction1_opt

# 无人机2飞行方向 (水平于xy平面)
drone_direction2_opt = np.array([np.cos(theta2_opt), np.sin(theta2_opt), 0])

# 投弹方向与无人机飞行方向一致
bomb_direction2_opt = drone_direction2_opt

# 无人机3飞行方向 (水平于xy平面)
drone_direction3_opt = np.array([np.cos(theta3_opt), np.sin(theta3_opt), 0])

# 投弹方向与无人机飞行方向一致
bomb_direction3_opt = drone_direction3_opt

# 导弹飞行方向
missile_direction = direction(M1, F)
v_missile = 300
total_missile_time = np.linalg.norm(M1 - F) / v_missile

# 时间参数
T = np.arange(0, total_missile_time, 0.01)
timeline = np.zeros(len(T))

count1, count2, count3 = 0, 0, 0
# 第一个烟幕弹
for i, t in enumerate(T):
    if count1 > 2000:
        break
    if t < A1_opt + B1_opt:
        continue

    # 计算烟幕弹位置
    n_jkt1 = np.array(FY1) + (A1_opt + B1_opt) * v_drone1_opt * bomb_direction1_opt - 0.5 * 9.8 * (
        B1_opt) ** 2 * np.array([0, 0, 1]) - 3 * (
                     t - A1_opt - B1_opt) * np.array([0, 0, 1])

    # 计算导弹位置
    M1_t = np.array(M1) + missile_direction * v_missile * t

    # 检验
    if test1(M1_t, n_jkt1):
        timeline[i] += 1
    count1 += 1

# 第二个烟幕弹
for i, t in enumerate(T):
    if count2 > 2000:
        break
    if t < A2_opt + B2_opt:
        continue

    # 计算烟幕弹位置
    n_jkt2 = np.array(FY2) + ( A2_opt + B2_opt) * v_drone2_opt * bomb_direction2_opt - 0.5 * 9.8 * (
        B2_opt) ** 2 * np.array(
        [0, 0, 1]) - 3 * (t -B2_opt - A2_opt) * np.array([0, 0, 1])

    # 计算导弹位置
    M1_t = np.array(M1) + missile_direction * v_missile * t

    # 检验
    if test1(M1_t, n_jkt2):
        timeline[i] += 1
    count2 += 1

# 第三个烟幕弹
for i, t in enumerate(T):
    if count3 > 2000:
        break
    if t < A3_opt + B3_opt:
        continue

    # 计算烟幕弹位置
    n_jkt3 = np.array(FY3) + ( A3_opt + B3_opt) * v_drone3_opt * bomb_direction3_opt - 0.5 * 9.8 * (
        B3_opt) ** 2 * np.array(
        [0, 0, 1]) - 3 * (t - B3_opt  - A3_opt) * np.array([0, 0, 1])

    # 计算导弹位置
    M1_t = np.array(M1) + missile_direction * v_missile * t

    # 检验
    if test1(M1_t, n_jkt3):
        timeline[i] += 1
    count3 += 1

# 确保值不超过1
timeline = np.clip(timeline, 0, 1)

total_time = np.sum(timeline) * 0.01

plt.figure(figsize=(10, 6))
plt.plot(T, timeline)
plt.xlabel('时间 (s)')
plt.ylabel('是否在目标区域')
plt.title('优化后炸弹在目标区域的时间')
plt.grid(True)
plt.show()

print(f"优化后在目标区域总时间: {total_time} 秒")

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(pso.fitness_history, 'b-', linewidth=2)
plt.title('PSO算法收敛曲线', fontsize=14)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('最佳适应度', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()