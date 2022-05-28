import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
import datetime
import numpy as np
import pdb
import matplotlib.pyplot as plt
import open3d as o3d

import os

def create_lineset(bbox, colors):
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

def load_view_point(pcd, filename, window_name,mode):
    if mode=='pred':
        left = 50
        top=50
    elif mode=='gt':
        left = 1000
        top=50
    else:
         print("model must be gt or pred")
         return
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
    # render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 3.0  # 设置渲染点的大小
    vis.run()
    img = vis.capture_screen_float_buffer(True)
    vis.destroy_window()

    return img

def load_pcd_bbox(mode,scan_name):
    pcd = o3d.io.read_point_cloud('vis_result2/' + mode + '/' + scan_name + '_pcd.ply')
    bbox_file = 'vis_result2/' + mode + '/' + scan_name + '_bbox.txt'
    bb = []
    with open(bbox_file,'r') as f:
        for line in f.readlines():
            points = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[0:8])]
            if mode == "gt":
                colors = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[9:12])]
            else:
                colors = [float(v) for i, v in enumerate(line.split("\n")[0].split(" ")[8:11])]
            bb.append(create_lineset(points, colors=colors))

    img = load_view_point([pcd], 'utils/viewpoint.json', window_name=scan_name + '_' + mode, mode=mode)
    plt.figure(dpi=300, figsize=(6, 4))

    # plt.subplot(1, 1, 1)
    # plt.title('pred')
    plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    plt.axis("off")

    plt.savefig('result/2.jpg')
    plt.show()

if __name__=='__main__':
    load_pcd_bbox('gt','scene0356_00')
