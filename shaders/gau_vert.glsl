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

layout (std430) buffer gaussian_pos {
    float g_pos[];
};
layout (std430) buffer gaussian_rot {
    float g_rot[];
};
layout(location = 3) in vec3 g_scale;
layout(location = 4) in vec3 g_dc_color;
layout(location = 5) in float g_opacity;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 hfovxy_focal;

out vec3 color;
out float alpha;
out vec3 conic;
out vec2 coordxy;  // local coordinate in quad, unit in pixel

vec3 computeColorFromSH(vec3 pos, vec3 campos, int deg, vec3[16] sh)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	vec3 dir = pos - campos;
    dir = normalize(dir);
	vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2_0 * xy * sh[4] +
				SH_C2_1 * yz * sh[5] +
				SH_C2_2 * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2_3 * xz * sh[7] +
				SH_C2_4 * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3_0 * y * (3.0f * xx - yy) * sh[9] +
					SH_C3_1 * xy * z * sh[10] +
					SH_C3_2 * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3_4 * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3_5 * z * (xx - yy) * sh[14] +
					SH_C3_6 * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
    return result;
}

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

    mat3 J = mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0
    );
    mat3 W = transpose(mat3(viewmatrix));
    mat3 T = W * J;

    mat3 cov = transpose(T) * transpose(cov3D) * T;
    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
    float scale_modifier = 1.f;
    vec4 gpos = vec4(g_pos[gl_InstanceID * 3], g_pos[gl_InstanceID * 3 + 1], g_pos[gl_InstanceID * 3 + 2], 1.f);
    vec4 grot = vec4(g_rot[gl_InstanceID * 4], g_rot[gl_InstanceID * 4 + 1], g_rot[gl_InstanceID * 4 + 2], g_rot[gl_InstanceID * 4 + 2]);
    // vec3 gscale = vec3(g_scale[gl_InstanceID * 3], g_scale[gl_InstanceID * 3 + 1], g_scale[gl_InstanceID * 3 + 2]);

    mat3 cov3d = computeCov3D(g_scale * scale_modifier, grot);
    vec4 g_pos_view = view_matrix * gpos;
    vec2 wh = 2 * hfovxy_focal.xy * hfovxy_focal.z;
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
    
    vec4 g_pos_screen = projection_matrix * g_pos_view;
    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));  // screen space half quad height and width
    vec2 quadwh_ndc = quadwh_scr / wh * 2;  // in ndc space
    g_pos_screen.xyz = g_pos_screen.xyz / g_pos_screen.w;
    g_pos_screen.w = 1.f;
    g_pos_screen.xy = g_pos_screen.xy + position * quadwh_ndc;
    coordxy = position * quadwh_scr;
    gl_Position = g_pos_screen;
    
    color = g_dc_color;
    alpha = g_opacity;
}
