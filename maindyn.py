import glfw
import OpenGL.GL as gl
from tqdm import tqdm
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from typing import List, Optional, Union
from renderer_cuda import CUDARenderer, gaus_cuda_from_cpu, GaussianDataCUDA


from util_gau import GaussianData
from copy import deepcopy

import subprocess


# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = util.Camera(720, 1280)
BACKEND_OGL = 0
BACKEND_CUDA = 1
g_renderer_list = [
    None,  # ogl
]
g_renderer_idx = BACKEND_OGL
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.0
g_auto_sort = False
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

# Function to load all .ply files from a directory
def load_all_ply_files(directory):
    print("[*] Loading PLY files...")
    all_gaussians = []
    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            file_path = os.path.join(directory, filename)
            try:
                gaussian_data = util_gau.load_ply(file_path)
                all_gaussians.append(gaussian_data)
            except RuntimeError as e:
                print(f"Failed to load {file_path}: {e}")
    print("[*] Done!")
    return all_gaussians

# Merge all the gaussians in the list and concatenate them
def merge_gaussians(gaussians: List[GaussianData], offsets: Optional[List[np.ndarray]] = None) -> GaussianData:
    print("[*] Merging PLY files...")
    xyz, rot, scale, opacity, sh = [], [], [], [], []
    if offsets is None:
        offsets = [np.array([0, 0, 0])] * len(gaussians)
    for g, offset in tqdm(zip(gaussians, offsets)):
        xyz.append(g.xyz + offset.astype(g.xyz.dtype))
        rot.append(g.rot)
        scale.append(g.scale)
        opacity.append(g.opacity)
        sh.append(g.sh)
    mergedGaussian = GaussianData(
        np.concatenate(xyz),
        np.concatenate(rot),
        np.concatenate(scale),
        np.concatenate(opacity),
        np.concatenate(sh)
    )
    print("[*] Done!")
    return mergedGaussian


class AnimatedGaussian:
    def __init__(self, renderer, frames_dir: str, animation_rate: int = 30):
        if isinstance(renderer, OpenGLRenderer):
            raise Exception("This animation dones't support the OpenGL renderer!")
        self.renderer = renderer
        self.original_frames = load_all_ply_files(frames_dir)
        self.active_gaussian_data_frames = self.to_cuda(deepcopy(self.original_frames))
        self.active_sh_deg_cache = []
        self.animation_rate = animation_rate
        self.frame_idx = 0
        self.draw_calls = 0
        frame = self.active_gaussian_data_frames[self.frame_idx]
        update_activated_renderer_state(frame)
        # TODO: Implement frame counting in a thread (+ mutex, argh)
        # 1. Spawn a thread with a function/method that counts the number of ms that elapsed
        # 2. If time_ms > self.animation_freq, then:
        # 3. write +1 to the frame_cnt
        # 4. In the main thread (draw method), read the frame_cnt!

    def to_cuda(self, frames: Union[GaussianData, List[GaussianData]]) -> List[GaussianDataCUDA]:
        if isinstance(frames, list):
            return [gaus_cuda_from_cpu(f) for f in frames]
        else:
            return gaus_cuda_from_cpu(frames)

    def draw(self):
        # print(f"draw_calls={self.draw_calls}, frame_idx={self.frame_idx}")
        self.draw_calls += 1
        if self.draw_calls == 0xffffff:
            self.draw_calls = 0
        if self.draw_calls % self.animation_rate == 0:
            # Change the frame
            self.frame_idx += 1
            if self.frame_idx == len(self.active_gaussian_data_frames):
                self.frame_idx = 0
            frame = self.active_gaussian_data_frames[self.frame_idx]
            sh_degree = self.active_sh_deg_cache[self.frame_idx] if len(self.active_gaussian_data_frames) == len(self.active_sh_deg_cache) else None
            # assert isinstance(frame, GaussianDataCUDA)
            self.renderer.update_gaussian_data(frame, sh_degree)

    def duplicate(self, n: int):
        try:
            random_offsets = [
                np.array([np.random.uniform(-10, 10), 0, np.random.uniform(-10, 10)]) for _ in range(n)
            ]
            self.active_gaussian_data_frames, self.active_sh_deg_cache = [], []
            for frame in self.original_frames:
                dup_frames = [frame] * n
                merged_gaussian_frames = self.to_cuda(merge_gaussians(dup_frames, offsets=random_offsets))
                self.active_gaussian_data_frames.append(merged_gaussian_frames)
                self.active_sh_deg_cache.append(self.renderer.compute_sh_degree(merged_gaussian_frames))
        except RuntimeError as e:
            print(f"Error merging and duplicating .ply files: {e}")

    def __len__(self):
        return len(self.active_gaussian_data_frames[self.frame_idx])


def impl_glfw_init():
    window_name = "Gaussian Crowd editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    global window
    window = glfw.create_window(g_camera.w, g_camera.h, window_name, None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
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

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, g_render_mode, g_render_mode_tables
    
    imgui.create_context()
    # Scale the entire UI
    io = imgui.get_io()
    io.font_global_scale = 2.5  # Adjust this value as needed

    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    gaussian_anim = None
    duplication_factor = 0
    update_activated_renderer_state(util_gau.naive_gaussian())
    # Settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        # update_camera_intrin_lazy()
        if gaussian_anim is not None:
            gaussian_anim.draw()
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["ogl", "cuda"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                changed, g_renderer.reduce_updates = imgui.checkbox("reduce updates", g_renderer.reduce_updates)

                # Display number of Gaussians
                imgui.text(f"# of Gaus = {len(gaussian_anim) if gaussian_anim is not None else 0}")
                if imgui.button(label='Open PLY'):
                    directory = filedialog.askdirectory(title="Open Directory",
                                                        initialdir="C:\\Users\\xiaoh\\Documents\\GaussianAvatarData\\PLY")
                    if directory:
                        try:
                            print("Loading Gaussian Animation...")
                            gaussian_anim = AnimatedGaussian(g_renderer, directory)
                        except RuntimeError as e:
                            print(f"Error loading .ply files: {e}")

                # Duplication factor slider
                changed, duplication_factor = imgui.slider_int("Duplication Factor", duplication_factor, 1, 2000) #change the numbers of individual
                if imgui.button(label='Crowd Generator'):
                    if gaussian_anim is not None:
                        gaussian_anim.duplicate(duplication_factor)

                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                # render mode
                changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)
                
                # sort button
                if imgui.button(label='sort Gaussians'):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox("auto sort", g_auto_sort)
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
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
                
                imgui.end()

        if g_show_camera_win:
            if imgui.begin("Camera Control", True):
                if imgui.button(label='rot 180'):
                    g_camera.flip_ground()

                changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
                if changed:
                    g_camera.update_target_distance()

                changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset r"):
                    g_camera.rot_sensitivity = 0.02

                changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset m"):
                    g_camera.trans_sensitivity = 0.01

                changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset z"):
                    g_camera.zoom_sensitivity = 0.01

                changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset ro"):
                    g_camera.roll_sensitivity = 0.03
                
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

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="Gaussian Crowd editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    # Start the MonitoringScript.py as a subprocess
    monitoring_process = subprocess.Popen(["python", "MonitoringScript.py"])

    try:
        # Run the main function
        main()
    except KeyboardInterrupt:
        print("Terminating the main script.")
    finally:
        # Terminate the monitoring subprocess if it's still running
        monitoring_process.terminate()
        monitoring_process.wait()
        print("Terminated MonitoringScript.py")
