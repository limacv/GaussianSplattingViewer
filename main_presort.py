import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import util_sort
import util_sort
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from dataclasses import make_dataclass

# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


g_width, g_height = 1280, 720
SortParams = make_dataclass("sortParams", ["num_draw", "method", "index", "normal"])
g_sort_params = SortParams(0, "distance", None, None)
g_camera = util.Camera(g_height, g_width)
g_program = None
g_scale_modifier = 1.
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_render_mode_tables = ["Normal", "Gaussian Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7
render_mode_offset = -4  # SH:0 always at index 0

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        int(g_width), int(g_height), window_name, None, None
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

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

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

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)

def main():
    global g_program, g_camera, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, \
        g_render_mode, g_render_mode_tables, g_sort_params
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # Load and compile shaders
    g_program = util.load_shaders('shaders/sort_gau_vert.glsl', 'shaders/gau_frag.glsl')

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
    util.set_uniform_1int(g_program, g_render_mode + render_mode_offset, "render_mod")
    util.set_uniform_1int(g_program, 1, "normal_cull")
    update_camera_pose()
    update_camera_intrin()
    
    # settings
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose()
        update_camera_intrin()
        
        gl.glUseProgram(g_program)
        gl.glBindVertexArray(vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, g_sort_params.num_draw)

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
                imgui.text(f"# of Gaus = {len(gaussians)}")
                imgui.text(f"# of Gaus Draw = {g_sort_params.num_draw}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="D:\\source\\data\\Gaussians\\output\\truck_mod12\\point_cloud\\iteration_30000",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            update_gaussian_data(gaussians)
                        except RuntimeError as e:
                            pass
                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    util.set_uniform_1f(g_program, g_scale_modifier, "scale_modifier")
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    util.set_uniform_1int(g_program, g_render_mode + render_mode_offset, "render_mod")
                
                # sort button
                imgui.text("Sort Method")
                if imgui.button(label='presort Gaus'):
                    g_sort_params.method = "depth"
                    g_sort_params.normal, g_sort_params.index = util_sort.presort_gaussian(gaussians, "depth")
                    update_index_buffer()
                if imgui.button(label='presort Gaus knn'):
                    g_sort_params.method = "knn"
                    g_sort_params.normal, g_sort_params.index = util_sort.presort_gaussian(gaussians, "knn")
                    update_index_buffer()
                if imgui.button(label='presort Gaus distance'):
                    g_sort_params.method = "distance"
                    g_sort_params.normal, g_sort_params.index = util_sort.presort_gaussian(gaussians, "distance")
                    update_index_buffer()
                if imgui.button(label='sort Gaussians'):
                    g_sort_params.method = "gt sort"
                    g_sort_params.index = util_sort.sort_gaussian(gaussians, g_camera.get_view_matrix())
                    g_sort_params.normal = None
                    update_index_buffer()
                if imgui.button(label='sort3'):
                    g_sort_params.method = "sort3"
                    util_sort.sort3_gaussian(gaussians)
                    g_sort_params.normal = None
                    g_auto_sort = True
                
                imgui.text(f"{g_sort_params.method} update")
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox(
                        "auto update", g_auto_sort,
                    )
                if g_auto_sort:
                    if g_sort_params.method == "gt sort":
                        g_sort_params.index = util_sort.sort_gaussian(gaussians, g_camera.get_view_matrix())
                        g_sort_params.normal = None
                        update_index_buffer()
                    elif g_sort_params.method == "sort3":
                        g_sort_params.index = util_sort.sort3_parse_index(g_camera.get_view_matrix())
                        update_index_buffer()
                    else:
                        update_index_buffer()
                
                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3
                    stride = nrChannels * width
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

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def update_index_buffer():
    global g_sort_params
    g_sort_params.num_draw = len(g_sort_params.index)
    if g_sort_params.normal is not None:
        util.set_uniform_1int(g_program, 1, "normal_cull")
        util.set_storage_buffer_data(g_program, "gaussian_normal", g_sort_params.normal, bind_idx=2)
    else:
        util.set_uniform_1int(g_program, 0, "normal_cull")
    util .set_storage_buffer_data(g_program, "gaussian_order", g_sort_params.index, bind_idx=1)


def update_gaussian_data(gaus: util_gau.GaussianDataBasic):
    # load gaussian geometry
    global g_num_gau_draw
    gaussian_data = gaus.flat()
    util.set_storage_buffer_data(g_program, "gaussian_data", gaussian_data, bind_idx=0)
    util.set_uniform_1int(g_program, gaus.sh_dim, "sh_dim")
    
    # presort configuration
    normal, index = util_sort.presort_gaussian(gaus)
    g_num_gau_draw = len(index)
    util.set_storage_buffer_data(g_program, "gaussian_order", index, bind_idx=1)
    util.set_storage_buffer_data(g_program, "gaussian_normal", normal, bind_idx=2)

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()
