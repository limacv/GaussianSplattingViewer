import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import time


g_camera = util.Camera(1000, 1000)
g_program = None
g_scale_modifier = 1.
g_auto_sort = False

def impl_glfw_init():
    width, height = 1000, 1000
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)


def update_camera_pose():
    if g_camera.is_pose_dirty:
        view_mat = g_camera.get_view_matrix()
        util.set_uniform_mat4(g_program, view_mat, "view_matrix")
        util.set_uniform_v3(g_program, g_camera.position, "cam_pos")
        g_camera.is_pose_dirty = False

def update_camera_intrin():
    if g_camera.is_intrin_dirty:
        proj_mat = g_camera.get_project_matrix()
        util.set_uniform_mat4(g_program, proj_mat, "projection_matrix")
        util.set_uniform_v3(g_program, g_camera.get_htanfovxy_focal(), "hfovxy_focal")
        g_camera.is_intrin_dirty = False


def main():
    global g_program, g_camera, g_scale_modifier, g_auto_sort
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)

    # Load and compile shaders
    g_program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

    # Vertex data for a quad
    quad_v = np.array([
        -1,  1,
        1,  1,
        1, -1,
        -1, -1
    ], dtype=np.float32).reshape(4, 2)
    quad_f = np.array([
        0, 1, 2,
        0, 2, 3
    ], dtype=np.uint32).reshape(2, 3)
    
    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_gaussian_data(gaussians)
    
    # load quad geometry
    vao, buffer_id = util.set_attributes(g_program, ["position"], [quad_v])
    util.set_faces_tovao(vao, quad_f)
    
    # set uniforms
    util.set_uniform_1f(g_program, g_scale_modifier, "scale_modifier")
    update_camera_pose()
    update_camera_intrin()
    
    # settings
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    fps = 40
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        start = time.time()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose()
        update_camera_intrin()
        
        gl.glUseProgram(g_program)
        gl.glBindVertexArray(vao)
        num_gau = len(gaussians)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

        # imgui ui
        if imgui.begin("Control", True):
            imgui.text(f"fps = {fps:.1f} frame / second")
            if imgui.button(label='open ply'):
                file_path = filedialog.askopenfilename(title="open ply",
                    initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                    filetypes=[('ply file', '.ply')]
                    )
                if file_path:
                    try:
                        gaussians = util_gau.load_ply(file_path)
                        update_gaussian_data(gaussians)
                    except RuntimeError as e:
                        pass
            
            changed, g_scale_modifier = imgui.slider_float(
                "scale modifier", g_scale_modifier, 0.1, 10, "scale = %.3f"
            )
            if changed:
                util.set_uniform_1f(g_program, g_scale_modifier, "scale_modifier")
            
            changed, g_camera.fovy = imgui.slider_float(
                "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
            )
            g_camera.is_intrin_dirty = changed
            update_camera_intrin()
            
            if imgui.button(label='sort Gaussians'):
                sort_gaussian(gaussians)
            imgui.same_line()
            changed, g_auto_sort = imgui.checkbox(
                    "auto sort", g_auto_sort,
                )
            if g_auto_sort:
                sort_gaussian(gaussians)
            
            if imgui.button(label='save image'):
                width, height = glfw.get_framebuffer_size(window)
                nrChannels = 3;
                stride = nrChannels * width;
                stride += (4 - stride % 4) if stride % 4 else 0
                gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                gl.glReadBuffer(gl.GL_FRONT)
                bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                imageio.imwrite("save.png", img[::-1])
                
                # save intermediate information
                np.savez(
                    "save.npz",
                    gau_xyz=gaussians.xyz,
                    gau_s=gaussians.scale,
                    gau_rot=gaussians.rot,
                    gau_c=gaussians.sh,
                    gau_a=gaussians.opacity,
                    viewmat=g_camera.get_view_matrix(),
                    projmat=g_camera.get_project_matrix(),
                    hfovxyfocal=g_camera.get_htanfovxy_focal()
                )
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
        
        end = time.time()
        fps_cur = 1 / (end - start) if end > start else fps
        fps = fps_cur * 1 / fps + fps * (1 - 1 / fps)

    impl.shutdown()
    glfw.terminate()


def sort_gaussian(gaus):
    xyz = gaus.xyz
    view_mat = g_camera.get_view_matrix()
    xyz_view = view_mat[None, :3, :3] @ xyz[..., None] + view_mat[None, :3, 3, None]
    depth = xyz_view[:, 2, 0]
    index = np.argsort(depth)
    index = index.astype(np.int32).reshape(-1, 1)
    util.set_storage_buffer_data(g_program, "gi", index, bind_idx=1)
    

def update_gaussian_data(gaus):
    # load gaussian geometry
    num_gau = len(gaus)
    gaussian_data = gaus.flat()
    util.set_storage_buffer_data(g_program, "gaussian_data", gaussian_data, bind_idx=0)
    indexing = np.arange(num_gau).astype(np.int32).reshape(-1, 1)
    indexing = np.random.permutation(num_gau).astype(np.int32).reshape(-1, 1)
    util.set_storage_buffer_data(g_program, "gi", indexing, bind_idx=1)
    util.set_uniform_1int(g_program, gaus.sh_dim, "sh_dim")

if __name__ == "__main__":
    main()
