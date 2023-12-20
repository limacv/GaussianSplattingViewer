from OpenGL import GL as gl
import util
import util_gau
import numpy as np



def _sort_gaussian(gaus: util_gau.GaussianData, view_mat):
    xyz = gaus.xyz
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    return index
    

class GaussianRenderBase:
    def __init__(self):
        self.gaussians = None

    def set_scale_modifier(self, modifier):
        raise NotImplementedError()
    
    def set_render_mod(self, mod: int):
        raise NotImplementedError()
    
    def update_camera_pose(self, camera: util.Camera):
        raise NotImplementedError()

    def update_camera_intrin(self, camera: util.Camera):
        raise NotImplementedError()
    
    def draw(self):
        raise NotImplementedError()
    
    def set_render_reso(self, w, h):
        raise NotImplementedError()


class OpenGLRenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        gl.glViewport(0, 0, w, h)
        self.program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

        # Vertex data for a quad
        self.quad_v = np.array([
            -1,  1,
            1,  1,
            1, -1,
            -1, -1
        ], dtype=np.float32).reshape(4, 2)
        self.quad_f = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32).reshape(2, 3)
        
        # load quad geometry
        vao, buffer_id = util.set_attributes(self.program, ["position"], [self.quad_v])
        util.set_faces_tovao(vao, self.quad_f)
        self.vao = vao

        # opengl settings
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.gaussians = gaus
        # load gaussian geometry
        gaussian_data = gaus.flat()
        util.set_storage_buffer_data(self.program, "gaussian_data", gaussian_data, bind_idx=0)
        util.set_uniform_1int(self.program, gaus.sh_dim, "sh_dim")

    def sort_and_update(self, camera: util.Camera):
        index = _sort_gaussian(self.gaussians, camera.get_view_matrix())
        util.set_storage_buffer_data(self.program, "gi", index, bind_idx=1)
        return
    
    def set_scale_modifier(self, modifier):
        util.set_uniform_1f(self.program, modifier, "scale_modifier")

    def set_render_mod(self, mod: int):
        util.set_uniform_1int(self.program, mod, "render_mod")

    def set_render_reso(self, w, h):
        gl.glViewport(0, 0, w, h)

    def update_camera_pose(self, camera: util.Camera):
        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(self.program, view_mat, "view_matrix")
        util.set_uniform_v3(self.program, camera.position, "cam_pos")

    def update_camera_intrin(self, camera: util.Camera):
        proj_mat = camera.get_project_matrix()
        util.set_uniform_mat4(self.program, proj_mat, "projection_matrix")
        util.set_uniform_v3(self.program, camera.get_htanfovxy_focal(), "hfovxy_focal")

    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        num_gau = len(self.gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(self.quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)
