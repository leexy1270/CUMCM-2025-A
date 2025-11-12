import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from test1 import test, direction
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 确保负号正常显示

#已知点坐标
True_obj=[0,200,0]
F=[0,0,0]
M1=[20000,0,2000]
M2=[19000,600,2100]
M3=[18000,-600,700]
FY1=[17800,0,1800]
FY2=[12000,1400,1400]
FY3=[6000,-3000,700]
FY4=[11000,2000,1800]
FY5=[13000,-2000,1300]

#参数
a=1.5
b=3.6
v_0=300
v_fy1=120


#时间段
A=np.arange(0,a,0.001)
B=np.arange(0,b,0.001)
T=np.arange(0,20,0.001)
t=a+b+T


D1=direction(M1,F) #导弹方向向量
D2=direction([17800,0,0],[0,0,0]) #无人机方向向量

#计算各时间点位置及测试
n_jkt=[]
M1_t=[]
result=[]
N=[]
for i in t:
    n_jkt=np.array(FY1)+(a+b)*v_fy1*D2-0.5*9.8*(b)**2*np.array([0,0,1])-3*(i-a-b)*np.array([0,0,1])
    M1_t=np.array(M1)+D1*v_0*i
    result.append(test(M1_t,n_jkt))
    N.append(n_jkt)  # 记录每个时间点的轰炸机位置

result=np.array(result)
fig=plt.figure(figsize=(10, 6))
plt.plot(T,result)
plt.xlabel('时间 (s)')
plt.grid(True)
plt.ylabel('是否在目标区域')
#plt.show()

print('有限遮蔽时间为：')
print(sum(result*(20/20000)))