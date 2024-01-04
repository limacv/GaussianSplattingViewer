#version 430 core

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(location = 0) in vec2 position;
// layout(location = 1) in vec3 g_pos;
// layout(location = 2) in vec4 g_rot;
// layout(location = 3) in vec3 g_scale;
// layout(location = 4) in vec3 g_dc_color;
// layout(location = 5) in float g_opacity;


#define POS_IDX 0
#define ROT_IDX 3
#define SCALE_IDX 7
#define OPACITY_IDX 10
#define SH_IDX 11

layout (std430, binding=0) buffer gaussian_data {
	float g_data[];
	// compact version of following data
	// vec3 g_pos[];
	// vec4 g_rot[];
	// vec3 g_scale[];
	// float g_opacity[];
	// vec3 g_sh[];
};
layout (std430, binding=1) buffer gaussian_order {
	int gi[];
};

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;
uniform vec3 cam_pos;
uniform int sh_dim;
uniform float scale_modifier;
uniform int render_mod;  // > 0 render 0-ith SH dim, -1 depth, -2 bill board, -3 gaussian
uniform int fisheye;

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel

mat3 computeCov3D(vec3 scale, vec4 q)  // should be correct
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    mat3 R = mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    mat3 M = S * R;
    mat3 Sigma = transpose(M) * M;
    return Sigma;
}

vec3 computeCov2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    // why need this? Try remove this later
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x / t.z;
    float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    mat3 J;
	if (fisheye > 0)
	{
		float eps = 0.01f;
		float x2 = t.x * t.x + eps;
		float y2 = t.y * t.y;
		float xy = t.x * t.y;
		float x2y2 = x2 + y2 ;
		float len_xy = length(t.xy) + eps;
		float x2y2z2_inv = 1.f / (x2y2 + t.z * t.z);

		float b = atan(len_xy, - t.z) / len_xy / x2y2;
		float a = t.z * x2y2z2_inv / (x2y2);
		J = mat3(
			focal_x * (x2 * a - y2 * b), focal_x * xy * (a + b),    - focal_x * t.x * x2y2z2_inv,
			focal_y * xy  * (a + b),    focal_y * (y2 * a - x2 * b), - focal_y * t.y * x2y2z2_inv,
			0, 0, 0
		);
	}
	else
	{
		J = mat3(
			focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
			0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
			0, 0, 0
		);
	} 
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

vec3 get_vec3(int offset)
{
	return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
	return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

void main()
{
	int boxid = gi[gl_InstanceID];
	int total_dim = 3 + 4 + 3 + 1 + sh_dim;
	int start = boxid * total_dim;
	vec4 g_pos = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view = view_matrix * g_pos;
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
	
	vec2 g_pos_screen;
	if (fisheye > 0)
	{
		float xy_len = length(g_pos_view.xy) + 0.0001f;
		float theta = atan(xy_len, - g_pos_view.z);
		if (abs(theta) > 3.14 * 0.403)  // 145 deg
			g_pos_screen = vec2(999, 999);
		else
			g_pos_screen = 2 * g_pos_view.xy * hfovxy_focal.z * theta / (xy_len * wh);
	}
	else
	{
		g_pos_screen = - 2 * hfovxy_focal.z * g_pos_view.xy / (g_pos_view.z * wh);
	}

	// early culling
	if (any(greaterThan(abs(g_pos_screen.xy), vec2(1.3))) || -g_pos_view.z < 0.01)
	{
		gl_Position = vec4(-100, -100, -100, 1);
		return;
	}
	vec4 g_rot = get_vec4(start + ROT_IDX);
	vec3 g_scale = get_vec3(start + SCALE_IDX);
	float g_opacity = g_data[start + OPACITY_IDX];

    mat3 cov3d = computeCov3D(g_scale * scale_modifier, g_rot);
    vec3 cov2d = computeCov2D(g_pos_view, 
                              hfovxy_focal.z, 
                              hfovxy_focal.z, 
                              hfovxy_focal.x, 
                              hfovxy_focal.y, 
                              cov3d, 
                              view_matrix);

    // Invert covariance (EWA algorithm)
	float det = (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
	if (det == 0.0f)
		gl_Position = vec4(0.f, 0.f, 0.f, 0.f);
    
    float det_inv = 1.f / det;
	conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    
    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // screen space half quad height and width
    vec2 quadwh_ndc = quadwh_scr / wh * 2;  // in ndc space
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = vec4(g_pos_screen, 0.f, 1.f);
    
    alpha = g_opacity;

	if (render_mod == -1)
	{
		float depth = -g_pos_view.z;
		depth = depth < 0.05 ? 1 : depth;
		depth = 1 / depth;
		color = vec3(depth, depth, depth);
		return;
	}

	// Covert SH to color
	int sh_start = start + SH_IDX;
	vec3 dir = g_pos.xyz - cam_pos;
    dir = normalize(dir);
	color = SH_C0 * get_vec3(sh_start);
	
	if (sh_dim > 3 && render_mod >= 1)  // 1 * 3
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		color = color - SH_C1 * y * get_vec3(sh_start + 1 * 3) + SH_C1 * z * get_vec3(sh_start + 2 * 3) - SH_C1 * x * get_vec3(sh_start + 3 * 3);

		if (sh_dim > 12 && render_mod >= 2)  // (1 + 3) * 3
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			color = color +
				SH_C2_0 * xy * get_vec3(sh_start + 4 * 3) +
				SH_C2_1 * yz * get_vec3(sh_start + 5 * 3) +
				SH_C2_2 * (2.0f * zz - xx - yy) * get_vec3(sh_start + 6 * 3) +
				SH_C2_3 * xz * get_vec3(sh_start + 7 * 3) +
				SH_C2_4 * (xx - yy) * get_vec3(sh_start + 8 * 3);

			if (sh_dim > 27 && render_mod >= 3)  // (1 + 3 + 5) * 3
			{
				color = color +
					SH_C3_0 * y * (3.0f * xx - yy) * get_vec3(sh_start + 9 * 3) +
					SH_C3_1 * xy * z * get_vec3(sh_start + 10 * 3) +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * get_vec3(sh_start + 11 * 3) +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * get_vec3(sh_start + 12 * 3) +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * get_vec3(sh_start + 13 * 3) +
					SH_C3_5 * z * (xx - yy) * get_vec3(sh_start + 14 * 3) +
					SH_C3_6 * x * (xx - 3.0f * yy) * get_vec3(sh_start + 15 * 3);
			}
		}
	}
	color += 0.5f;
}
