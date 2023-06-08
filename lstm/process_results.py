import os
import numpy as np
import numpy.typing as npt
import re
import open3d as o3d
from PIL import Image

from lstm.utils import SlicingDirection


def imgs_to_voi(input_dir: str, voi_shape: npt.NDArray, slicing_direction: SlicingDirection):
    image_names = sorted(os.listdir(input_dir))
    voi = np.zeros(voi_shape, dtype=int)

    img_dim = np.delete(voi_shape, slicing_direction.value)

    for img_name in image_names:
        img_path = os.path.join(input_dir, img_name)
    
        m = re.match(".*?level_([0-9]*?).png", img_name)

        if m is None:
            continue

        slice_level = int(m.group(1))
        img = Image.open(img_path).convert('1')
        img = img.resize(np.flip(img_dim))

        if slicing_direction == SlicingDirection.x:
            voi[slice_level, :, :] = img
        elif slicing_direction == SlicingDirection.y:
            voi[:, slice_level, :] = img
        else:
            voi[ :, :, slice_level] = img
    return voi


def voi_to_object_pcd(voi: npt.NDArray, color: npt.NDArray, visualize: bool = False):
    object = np.argwhere(voi != 0)
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(object)
    pcd.paint_uniform_color(color)
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def calculate_performance_metrics_from_pcds(dt_pcd: o3d.geometry.PointCloud, gt_pcd: o3d.geometry.PointCloud):
     # Remove duplicate points
    dt_pcd.remove_duplicated_points()
    gt_pcd.remove_duplicated_points()

    # Get voxel grid
    voxel_size = 1
    dt_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(dt_pcd,
                                                                voxel_size)

    gt_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(gt_pcd,
                                                                voxel_size=voxel_size)

    dt_included_in_gt = gt_voxel_grid.check_if_included(o3d.utility.Vector3dVector(np.asarray(dt_pcd.points)))
    gt_included_in_dt = dt_voxel_grid.check_if_included(o3d.utility.Vector3dVector(np.asarray(gt_pcd.points)))

    metrics = {}

    # TP = How many DT voxels are in GT
    metrics['TP'] = np.count_nonzero(dt_included_in_gt)
    # FP = How many DT voxels are not in GT
    metrics['FP'] = len(dt_included_in_gt) - metrics['TP']
    # FN = How many GT voxels are not in DT
    metrics['FN'] = len(gt_included_in_dt) - metrics['TP']
    # TN = all voxels - TP - FP - FN
    metrics['TN'] = 201*101*101 - metrics['TP'] - metrics['FP'] - metrics['FN']

    metrics['accuracy'] = (metrics['TP'] + metrics['TN']) / (metrics['TP'] + metrics['FP']+ metrics['FN'] + metrics['TN'])
    metrics['recall'] = metrics['TP'] / (metrics['TP'] + metrics['FN'])
    metrics['precision'] = metrics['TP'] / (metrics['TP'] + metrics['FP'])
    metrics['specificity'] = metrics['TN']/ (metrics['TN'] + metrics['FP'])

    return metrics