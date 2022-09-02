#!/usr/bin/env python3

import numpy as np
def normalize(u):
    return u / np.linalg.norm(u, axis = 1)[:, None]
def proj(u, v):
    return u * (np.sum(v * u, axis = 1) / np.sum(u * u, axis = 1))[:,None]
def ortho(u, v):
    return u - proj(v, u)
def angles_flex(vecs, rand=True):
    v1 = normalize(vecs[0] - vecs[1])
    v2 = normalize(vecs[2] - vecs[1])
    angle = np.arccos(np.sum(v1 * v2, axis = 1))
    if not rand:
        angle = np.rad2deg(angle)
    return angle
def angles_axis(vecs, rand=True):
    v1 = vecs[0] - vecs[1]
    v2 = vecs[1] - vecs[2]
    z = normalize(v1)
    x = normalize(ortho([1, 0, 0], z))
    y = np.cross(z, x)
    angle= np.arctan2(np.sum(v2 * y, axis = 1), np.sum(v2 * x, axis = 1))
    if not rand:
        angle = np.rad2deg(angle)
    return angle
def angles_crossaxis(vecs, rand=True):
    v1 = vecs[0] - vecs[1]
    v2 = vecs[1] -vecs[2]
    point = vecs[2] - vecs[0]
    z = normalize(np.cross(v1, v2))
    x = normalize(ortho([1, 0, 0], z))
    y = np.cross(z, x)
    angle = np.arctan2(np.sum(point * y, axis = 1), np.sum(point * x, axis = 1))
    if not rand:
        angle= np.rad2deg(angle)
    return angle

def compute_pose_angles(angles_keypoints_data, rand=True):
    angles = []
    angles.append(angles_flex(angles_keypoints_data,rand))
    angles.append(angles_axis(angles_keypoints_data,rand))
    angles.append(angles_crossaxis(angles_keypoints_data,rand))
    angles = np.array(angles).T
    return angles
