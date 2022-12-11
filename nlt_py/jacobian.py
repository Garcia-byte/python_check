import cv2
import numpy as np
import sys
import math

#旋转向量变旋转矩阵
def r_vec2matrix(rot_vec):
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    return rot_mat


def AngleAxisRotatePoint(angle_axis, pt):
    theta2 = np.dot(angle_axis, angle_axis)
    if (theta2 > sys.float_info.epsilon):
        theta = math.sqrt(theta2)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        theta_inverse = 1.0 / theta

        w = np.array([angle_axis[0] * theta_inverse,
                      angle_axis[1] * theta_inverse,
                      angle_axis[2] * theta_inverse])

        w_cross_pt = np.array([w[1] * pt[2] - w[2] * pt[1],
                               w[2] * pt[0] - w[0] * pt[2],
                               w[0] * pt[1] - w[1] * pt[0]])

        tmp = np.dot(w, pt) * ((1.0) - costheta)

        result = np.array([
            pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp,
            pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp,
            pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp,
        ])
    else:
        w_cross_pt = np.array([angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                               angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                               angle_axis[0] * pt[1] - angle_axis[1] * pt[0]])

        result = np.array([
            pt[0] + w_cross_pt[0],
            pt[1] + w_cross_pt[1],
            pt[2] + w_cross_pt[2],
        ])
    return result


def rotate(rot_vecs, points):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    # 参考: https://blog.csdn.net/qq_42658249/article/details/114494198
    #      https://zhuanlan.zhihu.com/p/113299607
    #      https://zhuanlan.zhihu.com/p/298128519
    # 旋转向量转换为旋转矩阵

    # 二范数
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


#相机成像过程，求雅可比矩阵每一个元素（求导）

def calcaJ(camera, point, six_or_nine):
    rot_mat = r_vec2matrix(camera[:3])
    pc = np.dot(rot_mat, point) + camera[3:6]
    # pc = AngleAxisRotatePoint(camera[0:3], point)

    x = -pc[0] / pc[2]
    y = -pc[1] / pc[2]

    f = camera[6]
    k0 = camera[7]
    k1 = camera[8]

    if six_or_nine:
        r2 = x * x + y * y

        uf = (1.0 + (k0 + k1 * r2) * r2) * x
        vf = (1.0 + (k0 + k1 * r2) * r2) * y

        uk0 = f * x * r2
        uk1 = f * x * r2 * r2

        vk0 = f * y * r2
        vk1 = f * y * r2 * r2

        ut0 = - f / pc[2] * (1.0 + k0 * r2 + 2 * k0 * x * x + k1 * r2 * r2 + 4.0 * k1 * r2 * x * x)
        ut1 = -f * x * y * (2 * k0 + 4 * k1 * r2) / pc[2]
        ut2 = -f / pc[2] * x * (1 + 3 * k0 * r2 + 5 * k1 * r2 * r2)

        vt0 = -f * x * y * (2 * k0 + 4 * k1 * r2) / pc[2]
        vt1 = -f / pc[2] * (1 + k0 * r2 + 2 * k0 * y * y + k1 * r2 * r2 + 4 * k1 * r2 * y * y)
        vt2 = -f / pc[2] * y * (1 + 3 * k0 * r2 + 5 * k1 * r2 * r2)

        uw0 = ut1 * (-np.dot(rot_mat[2], point)) + ut2 * (np.dot(rot_mat[1], point))
        uw1 = ut0 * (np.dot(rot_mat[2], point)) + ut2 * (-np.dot(rot_mat[0], point))
        uw2 = ut0 * (-np.dot(rot_mat[1], point)) + ut1 * (np.dot(rot_mat[0], point))

        vw0 = vt1 * (-np.dot(rot_mat[2], point)) + vt2 * (np.dot(rot_mat[1], point))
        vw1 = vt0 * (np.dot(rot_mat[2], point)) + vt2 * (-np.dot(rot_mat[0], point))
        vw2 = vt0 * (-np.dot(rot_mat[1], point)) + vt1 * (np.dot(rot_mat[0], point))

        ux = ut0 * rot_mat[0, 0] + ut1 * rot_mat[1, 0] + ut2 * rot_mat[2, 0]
        uy = ut0 * rot_mat[0, 1] + ut1 * rot_mat[1, 1] + ut2 * rot_mat[2, 1]
        uz = ut0 * rot_mat[0, 2] + ut1 * rot_mat[1, 2] + ut2 * rot_mat[2, 2]

        vx = vt0 * rot_mat[0, 0] + vt1 * rot_mat[1, 0] + vt2 * rot_mat[2, 0]
        vy = vt0 * rot_mat[0, 1] + vt1 * rot_mat[1, 1] + vt2 * rot_mat[2, 1]
        vz = vt0 * rot_mat[0, 2] + vt1 * rot_mat[1, 2] + vt2 * rot_mat[2, 2]

        return ut0, ut1, ut2, vt0, vt1, vt2, uw0, uw1, uw2, vw0, vw1, vw2, uf, uk0, uk1, \
               vf, vk0, vk1, ux, uy, uz, vx, vy, vz

#不考虑畸变时，只有6个参数的情况
    else:
        ut0 = -f / pc[2]
        ut1 = 0
        ut2 = -f * x / pc[2]

        vt0 = 0
        vt1 = -f / pc[2]
        vt2 = -f * y / pc[2]

        uw0 = ut2 * (np.dot(rot_mat[1], point))
        uw1 = ut0 * (np.dot(rot_mat[2], point)) + ut2 * (-np.dot(rot_mat[0], point))
        uw2 = ut0 * (-np.dot(rot_mat[1], point))

        vw0 = vt1 * (-np.dot(rot_mat[2], point)) + vt2 * (np.dot(rot_mat[1], point))
        vw1 = vt2 * (-np.dot(rot_mat[0], point))
        vw2 = vt1 * (np.dot(rot_mat[0], point))

        ux = ut0 * rot_mat[0, 0] + ut2 * rot_mat[2, 0]
        uy = ut0 * rot_mat[0, 1] + ut2 * rot_mat[2, 1]
        uz = ut0 * rot_mat[0, 2] + ut2 * rot_mat[2, 2]

        vx = vt1 * rot_mat[1, 0] + vt2 * rot_mat[2, 0]
        vy = vt1 * rot_mat[1, 1] + vt2 * rot_mat[2, 1]
        vz = vt1 * rot_mat[1, 2] + vt2 * rot_mat[2, 2]

        return ut0, ut1, ut2, vt0, vt1, vt2, uw0, uw1, uw2, vw0, vw1, vw2, 0, 0, 0, \
               0, 0, 0, ux, uy, uz, vx, vy, vz
