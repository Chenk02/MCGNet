# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Batch mode in loading Scannet scenes with vertices and ground truth labels
for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""
import os
import sys
import datetime
import time

import numpy as np
import pdb
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
import json
import pickle
import random
import scipy.io as sio
from utils.pc_util import params2bbox, write_ply_rgb

DATA_DIR = os.path.join('/home/ck/code/H3DNet/sunrgbd/sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_val') # path of sunrgbd dataset
VAL_SCAN_NAMES = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(DATA_DIR)])))
PRED_PATH= '/home/ck/code/H3DNet/eval_sunrgbd6/result' # path of predictions

def create_lineset_old(bbox, colors=[1, 0, 0]):
    ''' create bounding box
    '''
    xmin = bbox[0] - bbox[3] / 2
    xmax = bbox[0] + bbox[3] / 2
    ymin = bbox[1] - bbox[4] / 2
    ymax = bbox[1] + bbox[4] / 2
    zmin = bbox[2] - bbox[5] / 2
    zmax = bbox[2] + bbox[5] / 2
    points = [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmin], [xmin, ymax, zmax],
              [xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmin], [xmax, ymax, zmax]]
    lines = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4], [1, 5], [3, 7], [2, 6],
             [4, 5], [5, 7], [6, 7], [4, 6]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colors, [12, 1]))
    return line_set


def create_lineset(bbox, colors=[1, 0, 0]):
    ''' create bounding box
    '''
    points = params2bbox(bbox)
    lines = [[0, 1], [0, 2], [2, 3], [1, 3], [0, 4], [1, 5], [3, 7], [2, 6],
             [4, 5], [5, 7], [6, 7], [4, 6]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(colors, [12, 1]))
    return line_set


def load_view_point(pcd, filename, window_name):
    left = 50
    top=50

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=880, height=680, left=left, top=top)
    for part in pcd:
        vis.add_geometry(part)
    ctr = vis.get_view_control()
    current_param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = o3d.io.read_pinhole_camera_trajectory(filename)
    f = 983.80485869912241
    cx = current_param.intrinsic.width / 2 - 0.5
    cy = current_param.intrinsic.height / 2 - 0.5
    trajectory.parameters[0].intrinsic.set_intrinsics(current_param.intrinsic.width, current_param.intrinsic.height, f, f, cx, cy)

    ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.point_size = 1.0
    vis.run()
    img = vis.capture_screen_float_buffer(True)

    time.sleep(1)
    vis.destroy_window()
    return img

def select_bbox(bboxes):
    choose_ids = []
    for i in range(bboxes.shape[0]):
        if bboxes[i,-1] in OBJ_CLASS_IDS:
            choose_ids.append(i)
    bboxes = bboxes[choose_ids]
    return bboxes

def load_pcd_bbox(scan_name):
    pred_pcd = o3d.io.read_point_cloud('result/sunrgbd/vis_result/pred/' + scan_name + '_pcd.ply')
    # gt_pcd = o3d.io.read_point_cloud('result/sunrgbd/vis_result/gt/' + scan_name + '_pcd.ply')
    pred_bbox_file = 'result/sunrgbd/vis_result/pred/' + scan_name + '_bbox.txt'
    # gt_bbox_file = 'result/sunrgbd/vis_result/gt/' + scan_name + '_bbox.txt'
    pred_bb = []
    gt_bb = []
    with open(pred_bbox_file,'r') as f:
        for line in f.readlines():
            points = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[0:8])]
            colors = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[8:11])]
            pred_bb.append(create_lineset(points, colors=colors))

    # with open(gt_bbox_file,'r') as f2:
    #     for line in f2.readlines():
    #         points = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[0:8])]
    #         colors = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[8:11])]
    #         gt_bb.append(create_lineset(points, colors=colors))

    img_pred = load_view_point([pred_pcd] + pred_bb, 'utils/viewpoint.json', window_name=scan_name + "pred")
    # img_gt = load_view_point([gt_pcd] + gt_bb, 'utils/viewpoint.json', window_name=scan_name + "gt")

    # plt.figure(dpi=300,figsize=(6,4))

    # plt.subplot(1, 2, 1)
    # plt.title('pred')
    # plt.imshow(img_pred)
    # plt.xticks([])
    # plt.yticks([])

    # plt.subplot(1, 2, 2)
    # plt.title('gt')
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(img_gt)
    plt.imshow(img_pred)
    plt.axis('off')

    plt.savefig('result/img/sunrgbd_result/' + scan_name + '.png',dpi=500,bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    for i, scan_name in enumerate(VAL_SCAN_NAMES):
        # if not scan_name.endswith('10'):
        #     continue
        print('-' * 20 + 'begin')
        print(scan_name)
        load_pcd_bbox(scan_name)
        print('-' * 20 + 'done')

