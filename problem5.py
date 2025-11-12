import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from test1 import test1, direction

# 初始坐标
True_obj = np.array([0, 200, 0])
F = np.array([0, 0, 0])
M1 = np.array([20000, 0, 2000])
M2 = np.array([19000, 600, 2100])
M3 = np.array([18000, -600, 700])
FY1 = np.array([17800, 0, 1800])
FY2 = np.array([12000, 1400, 1400])
FY3 = np.array([6000, -3000, 700])
FY4 = np.array([11000, 2000, 1800])
FY5 = np.array([13000, -2000, 1300])
v_messile = 300

# 目标函数 - 计算有效遮蔽时间
def objective_function(x):
    # 解包参数: (无人机速度, 无人机航向角,投放时间*3,起爆时间*3)*5
    (v_drone1, theta1, A11, B11 ,A12, B12, A13, B13,
     v_drone2, theta2, A21, B21 ,A22, B22, A23, B23,
     v_drone3, theta3, A31, B31 ,A32, B32, A33, B33,
     v_drone4, theta4, A41, B41 ,A42, B42, A43, B43,
     v_drone5, theta5, A51, B51 ,A52, B52, A53, B53) = x



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

    # 无人机4飞行方向 (水平于xy平面)
    drone_direction4 = np.array([np.cos(theta4), np.sin(theta4), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction4 = drone_direction4

    # 无人机5飞行方向 (水平于xy平面)
    drone_direction5 = np.array([np.cos(theta5), np.sin(theta5), 0])

    # 投弹方向与无人机飞行方向一致
    bomb_direction5 = drone_direction5

    v_missile = 300

    # 时间参数
    total_missile_time = np.linalg.norm(M1 - F) / v_missile
    T = np.arange(0, total_missile_time, 0.01)

    # 计算结果 - 直接使用列表，不使用 pandas Series
    timeline = np.zeros(len(T))

    # 导弹1
    # 导弹飞行方向
    missile_direction1 = direction(M1, F)
    v_missile = 300  # 导弹速度300m/s

    # 无人机1
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A11 + B11:
            continue

        # 计算烟幕弹位置
        n_jkt11 = np.array(FY1) + (A11 + B11) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B11) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                         t - A11 - B11) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt11):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A11 + A12 + B12:
            continue

        # 计算烟幕弹位置
        n_jkt12 = np.array(FY1) + (A11 + A12 + B12) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B12) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B12 - A12) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt12):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A11 + A12 + A13 + B13:
            continue

        # 计算烟幕弹位置
        n_jkt13 = np.array(FY1) + (A11 + A12 + A13 + B13) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B13) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B13 - A12 - A13) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt13):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time11 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机2
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A21 + B21:
            continue

        # 计算烟幕弹位置
        n_jkt21 = np.array(FY2) + (A21 + B21) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B21) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                         t - A21 - B21) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt21):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A21 + A22 + B22:
            continue

        # 计算烟幕弹位置
        n_jkt22 = np.array(FY2) + (A21 + A22 + B22) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B22) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B22 - A22) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt22):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A21 + A22 + A23 + B23:
            continue

        # 计算烟幕弹位置
        n_jkt23 = np.array(FY2) + (A21 + A22 + A23 + B23) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (
            B23) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B23 - A22 - A23) * np.array([0, 0, 1])

        # 计算导弹位置
        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        # 检验
        if test1(M1_t, n_jkt23):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time12 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机3
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A31 + B31:
            continue

        n_jkt31 = np.array(FY3) + (A31 + B31) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B31) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B31) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt31):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A31 + A32 + B32:
            continue

        n_jkt32 = np.array(FY3) + (A31 + A32 + B32) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B32) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B32 - A32) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt32):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A31 + A32 + A33 + B33:
            continue

        n_jkt33 = np.array(FY3) + (A31 + A32 + A33 + B33) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (
            B33) ** 2 * np.array([0, 0, 1]) - 3 * (t - A31 - B33 - A32 - A33) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt33):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time13 = np.sum(timeline) * 0.01

    # 无人机4
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A41 + B41:
            continue

        n_jkt41 = np.array(FY4) + (A41 + B41) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B41) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B41) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt41):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A41 + A42 + B42:
            continue

        n_jkt42 = np.array(FY4) + (A41 + A42 + B42) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B42) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B42 - A42) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt42):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A41 + A42 + A43 + B43:
            continue

        n_jkt43 = np.array(FY4) + (A41 + A42 + A43 + B43) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (
            B43) ** 2 * np.array([0, 0, 1]) - 3 * (t - A41 - B43 - A42 - A43) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt43):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time14 = np.sum(timeline) * 0.01

    # 无人机5
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A51 + B51:
            continue

        n_jkt51 = np.array(FY5) + (A51 + B51) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B51) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B51) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt51):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A51 + A52 + B52:
            continue

        n_jkt52 = np.array(FY5) + (A51 + A52 + B52) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B52) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B52 - A52) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt52):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A51 + A52 + A53 + B53:
            continue

        n_jkt53 = np.array(FY5) + (A51 + A52 + A53 + B53) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (
            B53) ** 2 * np.array([0, 0, 1]) - 3 * (t - A51 - B53 - A52 - A53) * np.array([0, 0, 1])

        M1_t = np.array(M1) + missile_direction1 * v_missile * t

        if test1(M1_t, n_jkt53):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time15 = np.sum(timeline) * 0.01
    total_time1 = total_time15 + total_time14 + total_time13 + total_time12 + total_time11

    # 导弹2
    # 导弹飞行方向
    missile_direction2 = direction(M2, F)
    v_missile = 300  # 导弹速度300m/s

    # 初始化时间线数组
    timeline = np.zeros(len(T))

    # 无人机1
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A11 + B11:
            continue

        # 计算烟幕弹位置
        n_jkt11 = np.array(FY1) + (A11 + B11) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B11) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                          t - A11 - B11) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt11):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A11 + A12 + B12:
            continue

        # 计算烟幕弹位置
        n_jkt12 = np.array(FY1) + (A11 + A12 + B12) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B12) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B12 - A12) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt12):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A11 + A12 + A13 + B13:
            continue

        # 计算烟幕弹位置
        n_jkt13 = np.array(FY1) + (A11 + A12 + A13 + B13) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (
            B13) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B13 - A12 - A13) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt13):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time21 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机2
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A21 + B21:
            continue

        # 计算烟幕弹位置
        n_jkt21 = np.array(FY2) + (A21 + B21) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B21) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                          t - A21 - B21) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt21):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A21 + A22 + B22:
            continue

        # 计算烟幕弹位置
        n_jkt22 = np.array(FY2) + (A21 + A22 + B22) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B22) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B22 - A22) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt22):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A21 + A22 + A23 + B23:
            continue

        # 计算烟幕弹位置
        n_jkt23 = np.array(FY2) + (A21 + A22 + A23 + B23) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (
            B23) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B23 - A22 - A23) * np.array([0, 0, 1])

        # 计算导弹位置
        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        # 检验
        if test1(M2_t, n_jkt23):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time22 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机3
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A31 + B31:
            continue

        n_jkt31 = np.array(FY3) + (A31 + B31) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B31) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B31) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt31):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A31 + A32 + B32:
            continue

        n_jkt32 = np.array(FY3) + (A31 + A32 + B32) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B32) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B32 - A32) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt32):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A31 + A32 + A33 + B33:
            continue

        n_jkt33 = np.array(FY3) + (A31 + A32 + A33 + B33) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (
            B33) ** 2 * np.array([0, 0, 1]) - 3 * (t - A31 - B33 - A32 - A33) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt33):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time23 = np.sum(timeline) * 0.01

    # 无人机4
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A41 + B41:
            continue

        n_jkt41 = np.array(FY4) + (A41 + B41) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B41) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B41) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt41):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A41 + A42 + B42:
            continue

        n_jkt42 = np.array(FY4) + (A41 + A42 + B42) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B42) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B42 - A42) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt42):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A41 + A42 + A43 + B43:
            continue

        n_jkt43 = np.array(FY4) + (A41 + A42 + A43 + B43) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (
            B43) ** 2 * np.array([0, 0, 1]) - 3 * (t - A41 - B43 - A42 - A43) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt43):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time24 = np.sum(timeline) * 0.01

    # 无人机5
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A51 + B51:
            continue

        n_jkt51 = np.array(FY5) + (A51 + B51) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B51) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B51) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt51):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A51 + A52 + B52:
            continue

        n_jkt52 = np.array(FY5) + (A51 + A52 + B52) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B52) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B52 - A52) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt52):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A51 + A52 + A53 + B53:
            continue

        n_jkt53 = np.array(FY5) + (A51 + A52 + A53 + B53) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (
            B53) ** 2 * np.array([0, 0, 1]) - 3 * (t - A51 - B53 - A52 - A53) * np.array([0, 0, 1])

        M2_t = np.array(M2) + missile_direction2 * v_missile * t

        if test1(M2_t, n_jkt53):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time25 = np.sum(timeline) * 0.01
    total_time2 = total_time25 + total_time24 + total_time23 + total_time22 + total_time21

    # 导弹3
    # 导弹飞行方向
    missile_direction3 = direction(M3, F)
    v_missile = 300  # 导弹速度300m/s

    # 初始化时间线数组
    timeline = np.zeros(len(T))

    # 无人机1
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A11 + B11:
            continue

        # 计算烟幕弹位置
        n_jkt11 = np.array(FY1) + (A11 + B11) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B11) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                          t - A11 - B11) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt11):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A11 + A12 + B12:
            continue

        # 计算烟幕弹位置
        n_jkt12 = np.array(FY1) + (A11 + A12 + B12) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (B12) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B12 - A12) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt12):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A11 + A12 + A13 + B13:
            continue

        # 计算烟幕弹位置
        n_jkt13 = np.array(FY1) + (A11 + A12 + A13 + B13) * v_drone1 * bomb_direction1 - 0.5 * 9.8 * (
            B13) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A11 - B13 - A12 - A13) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt13):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time31 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机2
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A21 + B21:
            continue

        # 计算烟幕弹位置
        n_jkt21 = np.array(FY2) + (A21 + B21) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B21) ** 2 * np.array(
            [0, 0, 1]) - 3 * (
                          t - A21 - B21) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt21):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A21 + A22 + B22:
            continue

        # 计算烟幕弹位置
        n_jkt22 = np.array(FY2) + (A21 + A22 + B22) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (B22) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B22 - A22) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt22):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A21 + A22 + A23 + B23:
            continue

        # 计算烟幕弹位置
        n_jkt23 = np.array(FY2) + (A21 + A22 + A23 + B23) * v_drone2 * bomb_direction2 - 0.5 * 9.8 * (
            B23) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A21 - B23 - A22 - A23) * np.array([0, 0, 1])

        # 计算导弹位置
        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        # 检验
        if test1(M3_t, n_jkt23):
            timeline[i] += 1
        count3 += 1

    # 确保值不超过1
    timeline = np.clip(timeline, 0, 1)

    total_time32 = np.sum(timeline) * 0.01  # 每个时间点代表0.01秒

    # 无人机3
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A31 + B31:
            continue

        n_jkt31 = np.array(FY3) + (A31 + B31) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B31) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B31) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt31):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A31 + A32 + B32:
            continue

        n_jkt32 = np.array(FY3) + (A31 + A32 + B32) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (B32) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A31 - B32 - A32) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt32):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A31 + A32 + A33 + B33:
            continue

        n_jkt33 = np.array(FY3) + (A31 + A32 + A33 + B33) * v_drone3 * bomb_direction3 - 0.5 * 9.8 * (
            B33) ** 2 * np.array([0, 0, 1]) - 3 * (t - A31 - B33 - A32 - A33) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_messile * t

        if test1(M3_t, n_jkt33):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time33 = np.sum(timeline) * 0.01

    # 无人机4
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A41 + B41:
            continue

        n_jkt41 = np.array(FY4) + (A41 + B41) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B41) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B41) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt41):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A41 + A42 + B42:
            continue

        n_jkt42 = np.array(FY4) + (A41 + A42 + B42) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (B42) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A41 - B42 - A42) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt42):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A41 + A42 + A43 + B43:
            continue

        n_jkt43 = np.array(FY4) + (A41 + A42 + A43 + B43) * v_drone4 * bomb_direction4 - 0.5 * 9.8 * (
            B43) ** 2 * np.array([0, 0, 1]) - 3 * (t - A41 - B43 - A42 - A43) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt43):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time34 = np.sum(timeline) * 0.01

    # 无人机5
    count1, count2, count3 = 0, 0, 0
    # 第一个烟幕弹
    for i, t in enumerate(T):
        if count1 > 2000:
            break
        if t < A51 + B51:
            continue

        n_jkt51 = np.array(FY5) + (A51 + B51) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B51) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B51) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt51):
            timeline[i] += 1
        count1 += 1

    # 第二个烟幕弹
    for i, t in enumerate(T):
        if count2 > 2000:
            break
        if t < A51 + A52 + B52:
            continue

        n_jkt52 = np.array(FY5) + (A51 + A52 + B52) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (B52) ** 2 * np.array(
            [0, 0, 1]) - 3 * (t - A51 - B52 - A52) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt52):
            timeline[i] += 1
        count2 += 1

    # 第三个烟幕弹
    for i, t in enumerate(T):
        if count3 > 2000:
            break
        if t < A51 + A52 + A53 + B53:
            continue

        n_jkt53 = np.array(FY5) + (A51 + A52 + A53 + B53) * v_drone5 * bomb_direction5 - 0.5 * 9.8 * (
            B53) ** 2 * np.array([0, 0, 1]) - 3 * (t - A51 - B53 - A52 - A53) * np.array([0, 0, 1])

        M3_t = np.array(M3) + missile_direction3 * v_missile * t

        if test1(M3_t, n_jkt53):
            timeline[i] += 1
        count3 += 1

    timeline = np.clip(timeline, 0, 1)
    total_time35 = np.sum(timeline) * 0.01
    total_time3 = total_time35 + total_time34 + total_time33 + total_time32 + total_time31

    total_time = total_time3+total_time2+total_time1


    return -total_time

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


# 定义参数边界
# (无人机速度, 无人机航向角,投放时间*3,起爆时间*3)
bounds = [(70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (1, 7), (0.01, 7), (2, 7), (0.01, 7),
          (70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (1, 7), (0.01, 7), (2, 7), (0.01, 7),
          (70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (1, 7), (0.01, 7), (2, 7), (0.01, 7),
          (70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (1, 7), (0.01, 7), (2, 7), (0.01, 7),
          (70, 140), (0, 2 * np.pi), (0.01, 7), (0.01, 7), (1, 7), (0.01, 7), (2, 7), (0.01, 7)]

# 使用PSO算法进行优化
print("开始PSO优化...")
pso = PSO(objective_function, bounds, num_particles=60, max_iter=100)
xopt, fopt = pso.optimize()

print("PSO优化结果:")
print(
    f"最佳参数: ")
print(f"最大有效遮蔽时间: {-fopt} 秒")

# 使用最佳参数计算并可视化结果
# 输出最优参数
print("PSO优化结果:")
print("最佳参数:")
param_names = [
    "v_drone1", "theta1", "A11", "B11", "A12", "B12", "A13", "B13",
    "v_drone2", "theta2", "A21", "B21", "A22", "B22", "A23", "B23",
    "v_drone3", "theta3", "A31", "B31", "A32", "B32", "A33", "B33",
    "v_drone4", "theta4", "A41", "B41", "A42", "B42", "A43", "B43",
    "v_drone5", "theta5", "A51", "B51", "A52", "B52", "A53", "B53"
]

for i, (name, value) in enumerate(zip(param_names, xopt)):
    print(f"{name}: {value:.4f}")

print(f"最大有效遮蔽时间: {-fopt:.4f} 秒")

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(pso.fitness_history, 'b-', linewidth=2)
plt.title('PSO算法收敛曲线', fontsize=14)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('最佳适应度', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
