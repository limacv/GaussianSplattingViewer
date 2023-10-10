import numpy as np
from glm import *
import util

def computeCov3D(scale: vec3, q: vec4):
    S = mat3(0.)
    S[0][0] = scale.x
    S[1][1] = scale.y
    S[2][2] = scale.z
    r, x, y, z = q.x, q.y, q.z, q.w

    R = mat3(
		1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
		2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
		2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)
	);

    M = S * R;
    Sigma = transpose(M) * M;
    return Sigma;


def computeCov2D(mean_view: vec4, focal_x: float,  focal_y: float, tan_fovx: float, tan_fovy: float, cov3D: mat3, viewmatrix: mat4):
    t = mean_view;
    limx = 1.3 * tan_fovx;
    limy = 1.3 * tan_fovy;
    txtz = t.x / t.z;
    tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    J = mat3(
        focal_x / t.z, 0., -(focal_x * t.x) / (t.z * t.z),
		0., focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0
    );
    W = mat3(viewmatrix)
    T = W * J;

    cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3
    cov[1][1] += 0.3
    return vec3(cov[0][0], cov[0][1], cov[1][1])


if __name__ == "__main__":
    camera = util.Camera()
    view_matrix = camera.get_view_matrix()
    view_matrix = mat4(view_matrix)
    proj_matrix = camera.get_project_matrix()
    hfovxy_focal = camera.get_htanfovxy_focal()
    hfovxy_focal = vec3(*hfovxy_focal)

    g_pos = vec3(0, 0, 0)
    g_scale = vec3(1., 1., 1.)
    g_rot = vec4(1., 0, 0, 0)

    cov3d = computeCov3D(g_scale, g_rot)
    g_pos_view = view_matrix * vec4(g_pos, 1.)
    cov2d = computeCov2D(g_pos_view, 
                         hfovxy_focal.z, 
                         hfovxy_focal.z, 
                         hfovxy_focal.x, 
                         hfovxy_focal.y, 
                         cov3d, 
                         view_matrix)
    print(cov2d)

