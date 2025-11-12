# 检验函数 (保持原始代码中的逻辑)
import numpy as np

True_obj = np.array([0, 200, 0])

#视线遮挡测试函数
def test(M_position,n_jkt):
    direction_vec=direction(M_position,True_obj)
    #投影
    p_vec=np.array([direction_vec[0],direction_vec[1],0])
    p_vec=p_vec/np.linalg.norm(p_vec)
    p_vec_T=np.array([p_vec[1],-p_vec[0],0])
    #关键点
    key1=True_obj+p_vec*7+10*np.array([0,0,1])
    key2=True_obj-p_vec*7+10*np.array([0,0,1])
    key3=True_obj+p_vec_T*7+10*np.array([0,0,1])
    key4=True_obj-p_vec_T*7+10*np.array([0,0,1])
    key5=key3-5*np.array([0,0,1])
    key6=key4-5*np.array([0,0,1])
    key7=True_obj-p_vec*7
    key8=True_obj+p_vec_T*7
    key9=True_obj-p_vec_T*7
    keys=[key1,key2,key3,key4,key5,key6,key7,key8,key9]
    #检验
    for i in keys:
        s=np.dot((n_jkt-i),(M_position-i))/np.linalg.norm(M_position-i)**2
        s_=min(1,max(0,s))
        d2=np.linalg.norm(i+s_*(M_position-i)-n_jkt)**2
        if d2<=100:
            continue
        else:
            return 0
    return 1        

# 计算方向向量
def direction(A, B):
    AB = np.array(B) - np.array(A)
    return AB / np.linalg.norm(AB)

def test1(M_position, n_jkt):
    direction_vec = direction(M_position, True_obj)
    # 投影
    p_vec = np.array([direction_vec[0], direction_vec[1], 0])
    p_vec = p_vec / np.linalg.norm(p_vec)
    p_vec_T = np.array([p_vec[1], -p_vec[0], 0])
    # 关键点
    key1 = True_obj + p_vec * 7 + 10 * np.array([0, 0, 1])
    key2 = True_obj - p_vec * 7 + 10 * np.array([0, 0, 1])
    key3 = True_obj + p_vec_T * 7 + 10 * np.array([0, 0, 1])
    key4 = True_obj - p_vec_T * 7 + 10 * np.array([0, 0, 1])
    key5 = key3 - 5 * np.array([0, 0, 1])
    key6 = key4 - 5 * np.array([0, 0, 1])
    key7 = True_obj - p_vec * 7
    key8 = True_obj + p_vec_T * 7
    key9 = True_obj - p_vec_T * 7
    keys = [key1, key2, key3, key4, key5, key6, key7, key8, key9]
    # 检验
    for i in keys:
        s = np.dot((n_jkt - i), (M_position - i)) / np.linalg.norm(M_position - i) ** 2
        s_ = min(1, max(0, s))
        d2 = np.linalg.norm(i + s_ * (M_position - i) - n_jkt) ** 2
        if d2 <= 100:
            continue
        else:
            return 0
    return 1