from __future__ import print_function
import urllib
import urllib.request
import bz2
import os
import numpy as np
from numpy import mat
import cv2
import copy
import math

np.set_printoptions(suppress=True)

import pcl
import pcl.pcl_visualization
import random

from scipy.sparse import lil_matrix
from scipy.sparse import dia_matrix
import scipy.sparse
import matplotlib.pyplot as plt

import jacobian as ja


def vis_pair(cloud1, cloud2, rdm=False):
    color1 = [255, 0, 0]
    color2 = [0, 255, 0]
    if rdm:
        color1 = [255, 0, 0]
        color2 = [random.randint(0, 255) for _ in range(3)]
    visualcolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud1, color1[0], color1[1], color1[2])
    visualcolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud2, color2[0], color2[1], color2[2])
    vs = pcl.pcl_visualization.PCLVisualizering
    vss1 = pcl.pcl_visualization.PCLVisualizering()  # 初始化一个对象，这里是很重要的一步
    vs.AddPointCloud_ColorHandler(vss1, cloud1, visualcolor1, id=b'cloud', viewport=0)
    vs.AddPointCloud_ColorHandler(vss1, cloud2, visualcolor2, id=b'cloud1', viewport=0)
    vs.SetBackgroundColor(vss1, 0, 0, 0)
    # vs.InitCameraParameters(vss1)
    # vs.SetFullScreen(vss1, True)
    # v = True
    while not vs.WasStopped(vss1):
        vs.Spin(vss1)


BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/dubrovnik/"
# FILE_NAME = "problem-16-22106-pre-opt.txt.bz2"
# FILE_NAME = "problem-16-22106-pre-opt.txt.bz2"
FILE_NAME = "problem-16-22106-pre.txt.bz2"
# FILE_NAME = "problem-21-11315-pre.txt.bz2"
# FILE_NAME = "problem-39-18060-pre.txt.bz2"
# FILE_NAME = "problem-49-7776-pre.txt.bz2"
# FILE_NAME = "problem-50-20431-pre.txt.bz2"
URL = BASE_URL + FILE_NAME

if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)


def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())
        #.readline读取文件.split拆分字符串，通过指定分隔符对字符串进行切片，并返回分割后的字符串列表（list）
        print("相机pose数目(image数目) n_cameras: {}".format(n_cameras))
        print("重构出的3D点数目 n_points: {}".format(n_points))
        print("所有图像中2D特征点数目 n_observations: {}".format(n_observations))

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        # 读取每个特征点xy,及其对应的相机索引，重构的3D点索引
        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        # 读取每个相机的内参 0,1,2是R的旋转向量 3,4,5是平移向量
        # 6是焦距，7,8是畸变系数k1k2
        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        # 读取所有重构出的3D点，他们的list索引就是自身的索引
        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
# print(camera_params)
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

points_3d_pcl_ori = pcl.PointCloud(points_3d.astype(np.float32))
vis_pair(points_3d_pcl_ori, points_3d_pcl_ori)



JP_i = []

#计算雅可比矩阵，调用jacobian
def bundle_adjustment_sparsity(camera_indices, point_indices, camera_params, points_3d):
    # 计算雅可比矩阵比较麻烦，我们进行有限差分近似
    # 计算雅可比矩阵比较麻烦，我们进行有限差分近似
    # 构造雅可比稀疏结构(i. e. mark elements which are known to be non-zero)
    # 标记已知的非0元素
    JC = lil_matrix((m, 9 * n_cameras), dtype=float)
    JP = lil_matrix((m, 3 * n_points), dtype=float)

    jp_ele = np.mat([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    last_index_p = 0

    JP_i.clear()
    for s in range(camera_indices.size):
        index_c = camera_indices[s]
        index_p = point_indices[s]
        camera_sli = camera_params[camera_indices[s]]
        point_sli = points_3d[point_indices[s]]

        ut0, ut1, ut2, vt0, vt1, vt2, uw0, uw1, uw2, vw0, vw1, vw2, uf, uk0, uk1, \
        vf, vk0, vk1, ux, uy, uz, vx, vy, vz = \
            ja.calcaJ(camera_sli, point_sli, True)

        #JP关于三维点位置，JC关于相机参数
        JP[s * 2, index_p * 3 + 0] = ux
        JP[s * 2, index_p * 3 + 1] = uy
        JP[s * 2, index_p * 3 + 2] = uz
        JP[s * 2 + 1, index_p * 3 + 0] = vx
        JP[s * 2 + 1, index_p * 3 + 1] = vy
        JP[s * 2 + 1, index_p * 3 + 2] = vz

        JC[s * 2, index_c * 9 + 0] = uw0  # uw0
        JC[s * 2, index_c * 9 + 1] = uw1  # uw1
        JC[s * 2, index_c * 9 + 2] = uw2  # uw2
        JC[s * 2, index_c * 9 + 3] = ut0  # ut0
        JC[s * 2, index_c * 9 + 4] = ut1  # ut1
        JC[s * 2, index_c * 9 + 5] = ut2  # ut2
        JC[s * 2, index_c * 9 + 6] = uf  # uf
        JC[s * 2, index_c * 9 + 7] = uk0  # uk0
        JC[s * 2, index_c * 9 + 8] = uk1  # uk1
        JC[s * 2 + 1, index_c * 9 + 0] = vw0  # vw0
        JC[s * 2 + 1, index_c * 9 + 1] = vw1  # vw1
        JC[s * 2 + 1, index_c * 9 + 2] = vw2  # vw2
        JC[s * 2 + 1, index_c * 9 + 3] = vt0  # vt0
        JC[s * 2 + 1, index_c * 9 + 4] = vt1  # vt1
        JC[s * 2 + 1, index_c * 9 + 5] = vt2  # vt2
        JC[s * 2 + 1, index_c * 9 + 6] = vf  # vf
        JC[s * 2 + 1, index_c * 9 + 7] = vk0  # vk0
        JC[s * 2 + 1, index_c * 9 + 8] = vk1  # vk1

        if s == camera_indices.size - 1:
            jp_tmp = np.mat([[ux, uy, uz], [vx, vy, vz]])
            jp_ele = jp_ele + jp_tmp.transpose() * jp_tmp
            JP_i.append(copy.deepcopy(jp_ele))

        elif index_p == last_index_p:
            last_index_p = index_p
            jp_tmp = np.mat([[ux, uy, uz], [vx, vy, vz]])
            jp_ele = jp_ele + jp_tmp.transpose() * jp_tmp

        else:
            JP_i.append(copy.deepcopy(jp_ele))
            jp_ele = np.mat([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            last_index_p = index_p
            jp_tmp = np.mat([[ux, uy, uz], [vx, vy, vz]])
            jp_ele = jp_ele + jp_tmp.transpose() * jp_tmp

    return JP.tocsc(), JC.tocsc()

#手写cho
#def cholesky(n, S, V):
#    U =


class Adam():
    def __init__(self, g):
        self.B1 = 0.9
        self.B2 = 0.999
        self.e = 0.00000001
        self.mt = np.zeros(shape=g.shape)
        self.vt = np.zeros(shape=g.shape)
        self.B1_power = self.B1
        self.B2_power = self.B2

    def backward(self, g, WB, lr):
        self.mt = (self.B1 * self.mt).reshape(g.shape) + (1 - self.B1) * g
        self.vt = (self.B2 * self.vt).reshape(g.shape) + (1 - self.B2) * (g ** 2)
        mtt_w = self.mt / (1 - self.B1_power)
        vtt_w = self.vt / (1 - self.B2_power)
        # vtt_sqrt_w = np.array([math.sqrt(vtt_w[0]),
        #                        math.sqrt(vtt_w[1])])
        vtt_sqrt_w = np.sqrt(vtt_w)


        NWB = WB - (lr * mtt_w / (vtt_sqrt_w + self.e)).reshape(WB.shape)

        return NWB



#计算成本函数/残差（通过相机模型）
def calc_error(camera_indices, point_indices, camera_params, points_3d):
    E = lil_matrix((m, 1), dtype=float)
    for s in range(camera_indices.size):
        index_c = camera_indices[s]
        index_p = point_indices[s]
        camera = camera_params[camera_indices[s]]
        point = points_3d[point_indices[s]]
        rot_mat = ja.r_vec2matrix(camera[:3])
        pc = np.dot(rot_mat, point) + camera[3:6]

        #归一化
        x = -pc[0] / pc[2]
        y = -pc[1] / pc[2]

        f = camera[6]
        k0 = camera[7]
        k1 = camera[8]

        r2 = x * x + y * y
        d = 1.0 + r2 * (k0 + k1 * r2)

        p0 = f * d * x
        p1 = f * d * y         #通过相机模型计算出的观测值

        # E[s * 2, 0] = points_2d[s, 0] - p0
        # E[s * 2 + 1, 0] = points_2d[s, 1] - p1
        E[s * 2, 0] = p0 - points_2d[s, 0]
        E[s * 2 + 1, 0] = p1 - points_2d[s, 1]     #point_2d为真实像素值，无需计算

    return E.tocsc()     #返回稀疏矩阵的csc_matrix（压缩稀疏阵列）形式，乘加更高效


from scipy.sparse import dia_matrix    #对角储存形式
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
from scipy.sparse import identity
import time

error = calc_error(camera_indices, point_indices, camera_params, points_3d)
print("initial cost ", np.linalg.norm(error.todense()) ** 2 / 2)
plt.plot(error.todense())
plt.show()

start = time.time()

k = 0

#计算J矩阵
JP, JC = bundle_adjustment_sparsity(camera_indices, point_indices, camera_params, points_3d)
end1 = time.time()
print("jac1_runtime ", end1-start)
error = calc_error(camera_indices, point_indices, camera_params, points_3d)

AP = JP.T * JP
AC = JC.T * JC
#diagonal返回矩阵对角线元素
#计算Dp和Dc
IP = dia_matrix((AP.diagonal(), [0]), shape=(3 * n_points, 3 * n_points)).tocsc()
IC = dia_matrix((AC.diagonal(), [0]), shape=(9 * n_cameras, 9 * n_cameras)).tocsc()

# plt.plot(E.todense())
# plt.show()

v = 2
mu = (1e-15) * max(IP.max(), IC.max())

gp = JP.T * error
gc = JC.T * error

while k < 20:

    k += 1
    print(k)

    #舒尔消元法
    B = AC + mu * IC  # identity(9 * n_cameras) #IC
    E = JC.T * JP

    C_inverse = lil_matrix((3 * n_points, 3 * n_points), dtype=float)
    for i in range(n_points):
        JP_i[i][0, 0] += mu * IP[i * 3 + 0, i * 3 + 0]
        JP_i[i][1, 1] += mu * IP[i * 3 + 1, i * 3 + 1]
        JP_i[i][2, 2] += mu * IP[i * 3 + 2, i * 3 + 2]
        c_i = np.linalg.inv(JP_i[i])
        C_inverse[i * 3 + 0, i * 3 + 0] = c_i[0, 0]
        C_inverse[i * 3 + 0, i * 3 + 1] = c_i[0, 1]
        C_inverse[i * 3 + 0, i * 3 + 2] = c_i[0, 2]
        C_inverse[i * 3 + 1, i * 3 + 0] = c_i[1, 0]
        C_inverse[i * 3 + 1, i * 3 + 1] = c_i[1, 1]
        C_inverse[i * 3 + 1, i * 3 + 2] = c_i[1, 2]
        C_inverse[i * 3 + 2, i * 3 + 0] = c_i[2, 0]
        C_inverse[i * 3 + 2, i * 3 + 1] = c_i[2, 1]
        C_inverse[i * 3 + 2, i * 3 + 2] = c_i[2, 2]
    C_inverse = C_inverse.tocsc()

    # print("E[0, 0] ", E[0, 0])
    # print("E[143, 0] ", E[143, 0])
    # print("E[143, 66317] ", E[143, 66317])

    # ECI = E * C_inverse;
    # ECIET = ECI * E.T;
    #舒尔消元结果
    S = (B - E * C_inverse * E.T).todense()
    X = S.shape#.todense将稀疏矩阵转为稠密矩阵
    V = (-gc + E * C_inverse * gp).todense()
    delta_pc = np.linalg.solve(S, V)     #这里求解delta_pc直接用函数，但也可以用cholesky求解（硬件）
    delta_pp = (C_inverse * (-gp - E.T * delta_pc))
    d_pc = delta_pc.reshape(camera_params.shape)
    d_pp = delta_pp.reshape(points_3d.shape)
    Y_c = d_pc.shape
    Y_p = d_pp.shape
    print(Y_c, Y_p)
    # delta_pp = np.mat(spsolve(AP + mu * csc_matrix(np.identity(3*n_points)), gp)).T

    Delta_pc = Adam(np.zeros((16, 9)))
    Delta_pp = Adam(np.zeros((22106, 3)))



    #求P和dP的范数并比较
    norm_dp = math.sqrt(np.linalg.norm(delta_pp) ** 2 + np.linalg.norm(delta_pc) ** 2)
    norm_p = math.sqrt(np.linalg.norm(points_3d.reshape((3 * n_points, -1))) ** 2 + \
                       np.linalg.norm(camera_params.reshape((9 * n_cameras, -1))) ** 2)
    # if np.linalg.norm(np.vstack((delta_pp, delta_pc))) < (1e-12) * \
    #         np.linalg.norm(np.vstack((points_3d.reshape((3 * n_points, -1)),
    #                                   camera_params.reshape((9 * n_cameras, -1))))):
    #     break
    if norm_dp <= (1e-16) * norm_p:
        break
    else:
        #得到Pnew
        pc = camera_params
        pp = points_3d
        for i in range(n_cameras):
            d_pc[i, 0] += delta_pc[9 * i + 0, 0]
            d_pc[i, 1] += delta_pc[9 * i + 1, 0]  # 1
            d_pc[i, 2] += delta_pc[9 * i + 2, 0]  # 2
            d_pc[i, 3] += delta_pc[9 * i + 3, 0]  # 3
            d_pc[i, 4] += delta_pc[9 * i + 4, 0]  # 4
            d_pc[i, 5] += delta_pc[9 * i + 5, 0]  # 5
            d_pc[i, 6] += delta_pc[9 * i + 6, 0]  # 6
            d_pc[i, 7] += delta_pc[9 * i + 7, 0]  # 7
            d_pc[i, 8] += delta_pc[9 * i + 8, 0]  # 8

        for i in range(n_points):
            d_pp[i, 0] += delta_pp[3 * i + 0, 0]
            d_pp[i, 1] += delta_pp[3 * i + 1, 0]
            d_pp[i, 2] += delta_pp[3 * i + 2, 0]

        pc = np.asarray(pc)
        pp = np.asarray(pp)
        d_pc = np.asarray(d_pc)
        d_pp = np.asarray(d_pp)

        pc_new = pc + Delta_pc.backward(d_pc, pc, 0.001)
        pp_new = pp + Delta_pp.backward(d_pp, pp, 0.0001)

        # for i in range(n_cameras):
        #     pc_new[i, 0] += delta_pc[9 * i + 0, 0]  # 0
        #     pc_new[i, 1] += delta_pc[9 * i + 1, 0]  # 1
        #     pc_new[i, 2] += delta_pc[9 * i + 2, 0]  # 2
        #     pc_new[i, 3] += delta_pc[9 * i + 3, 0]  # 3
        #     pc_new[i, 4] += delta_pc[9 * i + 4, 0]  # 4
        #     pc_new[i, 5] += delta_pc[9 * i + 5, 0]  # 5
        #     pc_new[i, 6] += delta_pc[9 * i + 6, 0]  # 6
        #     pc_new[i, 7] += delta_pc[9 * i + 7, 0]  # 7
        #     pc_new[i, 8] += delta_pc[9 * i + 8, 0]  # 8
        #
        # for i in range(n_points):
        #     pp_new[i, 0] += delta_pp[3 * i + 0, 0]
        #     pp_new[i, 1] += delta_pp[3 * i + 1, 0]
        #     pp_new[i, 2] += delta_pp[3 * i + 2, 0]



        e_new = calc_error(camera_indices, point_indices, pc_new, pp_new).todense()
        print(error-e_new)
        if np.linalg.norm(error-e_new) ** 2 < 1.0e-7:
            break

        n1 = np.linalg.norm(error.todense()) ** 2
        n2 = np.linalg.norm(e_new) ** 2
        g = np.mat(np.vstack((gc.todense(), gp.todense())))
        delta_p = np.mat(np.vstack((delta_pc, delta_pp)))
        rho = (n1 - n2) / (delta_p.T * (mu * delta_p - g))

        #rou大于0更新，小于0扩大信任域
        if rho[0, 0] > 0:

            camera_params = pc_new
            points_3d = pp_new

            JP, JC = bundle_adjustment_sparsity(camera_indices, point_indices, camera_params, points_3d)
            error = calc_error(camera_indices, point_indices, camera_params, points_3d)

            # plt.plot(error.todense())

            AP = JP.T * JP
            AC = JC.T * JC
            IP = dia_matrix((AP.diagonal(), [0]), shape=(3 * n_points, 3 * n_points)).tocsc()
            IC = dia_matrix((AC.diagonal(), [0]), shape=(9 * n_cameras, 9 * n_cameras)).tocsc()

            gp = JP.T * error
            gc = JC.T * error

            if np.linalg.norm(g, ord=np.Inf) <= (1e-16):
                break

            # print(1 - pow((2 * (rho[0, 0]) - 1), 3))
            mu = mu * max(1 / 3, 1 - pow(2 * rho[0, 0] - 1, 3))

            v = 2

        else:
            mu = mu * v
            v = 2 * v

# for i in range(n_cameras):
#     camera_params[i, 0] += delta_pc[9 * i + 0, 0]
#     camera_params[i, 1] += delta_pc[9 * i + 1, 0]
#     camera_params[i, 2] += delta_pc[9 * i + 2, 0]
#     camera_params[i, 3] += delta_pc[9 * i + 3, 0]
#     camera_params[i, 4] += delta_pc[9 * i + 4, 0]
#     camera_params[i, 5] += delta_pc[9 * i + 5, 0]
#     camera_params[i, 6] += delta_pc[9 * i + 6, 0]
#     camera_params[i, 7] += delta_pc[9 * i + 7, 0]
#     camera_params[i, 8] += delta_pc[9 * i + 8, 0]
#
# for i in range(n_points):
#     points_3d[i, 0] += delta_pp[3 * i + 0, 0]
#     points_3d[i, 1] += delta_pp[3 * i + 1, 0]
#     points_3d[i, 2] += delta_pp[3 * i + 2, 0]

end = time.time()

error = calc_error(camera_indices, point_indices, camera_params, points_3d)
print("final cost", np.linalg.norm(error.todense()) ** 2 / 2)
plt.plot(error.todense())
plt.show()
print("runtime ", end-start)

points_3d_pcl_target = pcl.PointCloud(points_3d.astype(np.float32))
vis_pair(points_3d_pcl_target, points_3d_pcl_target)
vis_pair(points_3d_pcl_ori, points_3d_pcl_target)
