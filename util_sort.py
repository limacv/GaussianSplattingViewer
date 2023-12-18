import numpy as np
from util_gau import GaussianDataBasic
import open3d
import util_gau


def sort_gaussian(gaus: util_gau.GaussianDataBasic, view_mat):
    xyz = gaus.xyz
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index


def knn_smoothed_normal(gaus: GaussianDataBasic):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(gaus.xyz)         
    pcd_tree = open3d.geometry.KDTreeFlann(pcd)
    idx = [pcd_tree.search_knn_vector_3d(p, knn=100)[1] for p in pcd.points]
    idx = np.array(idx)
    normals = gaus.normal[idx]
    normals_mean = np.mean(normals, axis=-2)
    normals_mean = normals_mean / np.linalg.norm(normals_mean, axis=-1, keepdims=True)
    return np.ascontiguousarray(normals_mean).astype(np.float32)


def knn_normal(gaus: GaussianDataBasic):
    # could substitute normal computation on-the-fly
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(gaus.xyz)
    pcd.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(1000))
    pcd.orient_normals_consistent_tangent_plane(1)
    normal = np.asarray(pcd.normals).astype(np.float32)
    return normal


def presort_gaussian(gaus: GaussianDataBasic, normal_mod: str="depth"):
    pos = gaus.xyz
    center = np.mean(pos, axis=0)
    pos_recenter = pos - center[None]
    
    if normal_mod == "depth":
        normal = gaus.normal
    elif normal_mod == "smoothed":
        normal = knn_smoothed_normal(gaus)
    elif normal_mod == "knn":
        normal = knn_normal(gaus)
    elif normal_mod == "distance":
        normal = pos_recenter / np.linalg.norm(pos_recenter, axis=-1, keepdims=True)
    else:
        raise RuntimeError(f"Unrecognized normal mode {normal_mod}")
    
    normal_all = np.concatenate([normal, -normal])  # first len(pos) are original, second len(pos) are copys
    pos_recenter = np.concatenate([pos_recenter, pos_recenter])
    dot = (normal_all * pos_recenter).sum(axis=-1)
    index = np.argsort(dot)
    normal_all = normal_all[index]    
    index = index % len(pos)
    
    normal = np.ascontiguousarray(normal_all).astype(np.float32).reshape(-1, 3)
    index = index.astype(np.int32).reshape(-1, 1)
    return normal, index


g_sort3_indexes = []
def sort3_gaussian(gaus: util_gau.GaussianDataBasic):
    global g_sort3_indexes
    xyz = gaus.xyz
    index_x = np.argsort(xyz[:, 0]).astype(np.int32).reshape(-1, 1)
    index_y = np.argsort(xyz[:, 1]).astype(np.int32).reshape(-1, 1)
    index_z = np.argsort(xyz[:, 2]).astype(np.int32).reshape(-1, 1)

    g_sort3_indexes = [index_x, index_y, index_z]

def sort3_parse_index(viewmat):
    viewdir = viewmat[2, :3]
    idx = np.argmax(np.abs(viewdir))
    index = g_sort3_indexes[idx]
    index = index if viewdir[idx] > 0 else index[::-1]
    return np.ascontiguousarray(index)


if __name__ == "__main__":
    from util_gau import naive_gaussian
    gaus = naive_gaussian()
    knn_smoothed_normal(gaus)
