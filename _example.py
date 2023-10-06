import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
from util import *
import trimesh
import glm
import imageio
import tkinter as tk
from tkinter import filedialog
import os
import time

background_color = (35 / 255, 35 / 255, 35 / 255, 1)
# background_color = (1, 1, 1, 1)
# camera parameter
window_h, window_w = 0, 0
camera_fov = 15
camera_fov_speed = 1
camera_pos = glm.vec3(0, 0, -7)
camera_tar = glm.vec3(0, 0, 0)
camera_up = glm.vec3(0, 1, 0)
camera_near = 0.1
camera_far = 100

# camera control
last_time = 0
mouse_is_moving_camera = False
click_pos = 0, 0
click_camera_pos = None
click_camera_right = None
click_camera_up = None
mouse_speed = 0.02
keyboard_speed = 1

# point control
mouse_is_moving_point = False
cpts = None
cpts_seleted_idx = None
cpts_seleted_pos = None
cpts_isdirty = True
cpts_seleted_threshold = 20
camera_fx, camera_fy = 0, 0

# imgui
show_control_panel = True
show_metric_window = False
show_texture_map = True
head_transparency = 1
render_head = True
control_transparency = 1
control_size = 1
render_control = True

# appearance editing
pen = None
pen_size = 5
pen_color = (0, 0, 0, 1)

# file
texture_file_path = "M:\\NeRFtx\\ablation"  # initial folder
obj_file_path = "M:\\NeRFtx\\ablation"
cpts_file_path = os.path.dirname(obj_file_path) + "/cpoint.npz"

# output
render_success = False
render_success_time = 0


def impl_glfw_init():
    width, height = 1280, 720
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
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def my_process_inputs():
    global camera_pos, camera_tar, camera_up, last_time
    current_time = glfw.get_time()
    delta_time = current_time - last_time
    last_time = current_time
    cameraSpeed = delta_time * keyboard_speed
    if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS):
        camera_pos += cameraSpeed * (camera_tar - camera_pos)
    elif (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS):
        camera_pos -= cameraSpeed * (camera_tar - camera_pos)
    # elif (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
    #     radius = glm.length(camera_pos)
    #     camera_pos -= glm.normalize(glm.cross((camera_tar - camera_pos), camera_up)) * cameraSpeed
    #     camera_pos = glm.normalize(camera_pos) * radius
    # elif (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
    #     radius = glm.length(camera_pos)
    #     camera_pos += glm.normalize(glm.cross((camera_tar - camera_pos), camera_up)) * cameraSpeed
    #     camera_pos = glm.normalize(camera_pos) * radius
    # elif (glfw.get_key(window, glfw.KEY_E) == glfw.PRESS):
    #     camera_tar[2] += cameraSpeed
    # elif (glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS):
    #     camera_tar[2] -= cameraSpeed


def wheel_callback(window, dx, dy):
    global camera_fov, camera_fov_speed
    camera_fov += (camera_fov_speed * dy)


def mouse_callback(window, button, action, mods):
    global mouse_is_moving_camera, click_pos, camera_pos, camera_tar, camera_up, \
        click_camera_pos, click_camera_right, click_camera_up, \
        cpts, cpts_seleted_idx, mouse_is_moving_point, cpts_seleted_pos, camera_fx, camera_fy

    io = imgui.get_io()
    if io.want_capture_mouse:
        return

    global render_control

    if (button == glfw.MOUSE_BUTTON_LEFT):
        if action == glfw.PRESS:
            click_pos = glfw.get_cursor_pos(window)
            click_camera_pos = camera_pos
            click_camera_right = glm.normalize(glm.cross(camera_up, camera_pos - camera_tar))
            click_camera_up = glm.normalize(glm.cross(click_camera_right, camera_pos - camera_tar))

            # check whether hit any point
            view_mat, proj_mat = get_view_proj_mat()
            cpts_2d = np.concatenate([cpts, np.ones_like(cpts[:, :1])], -1)
            cpts_2d = (proj_mat @ view_mat @ cpts_2d.T)
            cpts_2d[0] = -cpts_2d[0]
            cpts_2d_norm = (cpts_2d[:2] / cpts_2d[3:]) + 1
            dnorm = np.array([window_w / 2, window_h / 2])[:, None]
            cpts_2d_screen = cpts_2d_norm * dnorm

            dist = np.linalg.norm(cpts_2d_screen - np.array([click_pos[0], click_pos[1]])[:, None], axis=0)
            cpts_seleted_idx = np.argmin(dist)

            if render_control and dist[cpts_seleted_idx] < cpts_seleted_threshold * control_size:
                mouse_is_moving_point = True
                cpts_seleted_pos = cpts[cpts_seleted_idx].copy()
                camera_fx = 2 / window_w * cpts_2d[3, cpts_seleted_idx] / proj_mat[0, 0]
                camera_fy = 2 / window_h * cpts_2d[3, cpts_seleted_idx] / proj_mat[1, 1]
                mouse_is_moving_camera = False
            else:
                mouse_is_moving_camera = True
                mouse_is_moving_point = False

        elif action == glfw.RELEASE:
            mouse_is_moving_camera = False
            mouse_is_moving_point = False


def cursor_pos_callback(window, xpos, ypos):
    global camera_pos, click_camera_pos, click_camera_right, click_camera_up, mouse_is_moving_camera, \
        cpts, cpts_seleted_idx, cpts_isdirty, cpts_seleted_pos, camera_fx, camera_fy
    if mouse_is_moving_camera:
        dx, dy = (xpos - click_pos[0]) * mouse_speed, (ypos - click_pos[1]) * mouse_speed
        radius = glm.length(click_camera_pos)
        camera_pos = click_camera_pos + (dx * click_camera_right + dy * click_camera_up)
        camera_pos = glm.normalize(camera_pos) * radius
    elif mouse_is_moving_point:
        dx, dy = (click_pos[0] - xpos) * camera_fx, (click_pos[1] - ypos) * camera_fy
        cpts[cpts_seleted_idx] = cpts_seleted_pos + (dx * click_camera_right + dy * click_camera_up)
        cpts_isdirty = True


def get_view_proj_mat():
    project_mat = glm.perspective(
        glm.radians(camera_fov),
        window_w / window_h if window_h > 0 else 0,
        camera_near,
        camera_far
    )
    project_mat = np.array(project_mat).astype(np.float32)
    view_mat = glm.lookAt(camera_pos, camera_tar, camera_up)
    view_mat = np.array(view_mat).astype(np.float32)
    return view_mat, project_mat


def get_camera_pose_for_nerf():
    view_mat = glm.lookAt(camera_pos, camera_tar, camera_up)
    view_mat = np.array(view_mat).astype(np.float32)
    pose = np.linalg.inv(view_mat)
    pose[:, 0] = -pose[:, 0]
    pose[:, 2] = -pose[:, 2]
    # pose[0] = -pose[0]
    intrinsic = np.zeros_like(pose[:3, :3])
    fx = fy = 0.5 / np.tan(glm.radians(camera_fov) / 2)
    cx, cy = 0.5, 0.5
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    intrinsic[2, 2] = 1
    return pose, intrinsic


def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    root = tk.Tk()  # used for file dialog
    root.withdraw()
    glfw.set_mouse_button_callback(window, mouse_callback);
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    print(glGetString(GL_VERSION))

    head_shader = load_shaders('shaders/vertex.glsl', 'shaders/frag.glsl')
    control_shader = load_shaders('shaders/cvertex.glsl', 'shaders/cfrag.glsl')

    # anything about head
    global obj_file_path, texture_file_path, cpts_file_path

    while True:
        new_obj_file = filedialog.askopenfilename(
            title="open mesh",
            initialdir=obj_file_path,
            filetypes=[('mesh file', '.obj')])
        working_dir = os.path.dirname(new_obj_file)
        obj_file_path = new_obj_file
        texture_file_path = os.path.join(working_dir, "texture.png")
        cpts_file_path = os.path.join(working_dir, "cpoint.npz")
        if os.path.exists(texture_file_path) and os.path.exists(cpts_file_path):
            break

    mesh = trimesh.load(obj_file_path)
    texture = imageio.imread(texture_file_path)[..., :3]
    verts = mesh.vertices.astype(np.float32)
    verts = np.concatenate([verts, np.ones_like(verts[:, :1])], -1)
    faces = mesh.faces.astype(np.int32)
    faces = faces[:, ::-1].copy()
    uvs = mesh.visual.uv.astype(np.float32)
    uvs[:, 1] = (1 - uvs[:, 1])
    vao, bufferids = set_attributes(head_shader, ['position', 'uv'], [verts, uvs])
    set_faces_tovao(vao, faces)

    texture_id = set_texture2d(texture)
    set_uniform_1int(head_shader, 0, "tex")  # 0 is the GL_TEXTURE0

    # anything about control points
    global cpts, cpts_isdirty, cpts_seleted_idx
    cpts_file = np.load(cpts_file_path)
    frame_idx = cpts_file["frameidx"]
    cpts = cpts_file["cpts"][frame_idx].astype(np.float32)
    cpts_w = cpts_file["radius"].astype(np.float32)
    cpts_w = cpts_w[frame_idx] if len(cpts_w) > 1 else cpts_w[0]
    cpts_canonical = cpts_file["cano"][0]
    transform = cpts_file["trans"][frame_idx]
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]])]).astype(np.float32)
    transform_inv = np.linalg.inv(transform)
    # transform the cpts to observation space
    cpts = np.concatenate([cpts, np.ones_like(cpts[:, :1])], -1) @ transform_inv.T
    cpts = cpts[:, :3]
    set_uniform_mat4(head_shader, transform, "global_trans")

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.02, color=None)
    cverts = sphere.vertices.astype(np.float32)
    cfaces = sphere.faces.astype(np.int32)
    # cfaces = cfaces[:, ::-1].copy()
    cvao, cbufferids = set_attributes(control_shader, ['position'], [cverts])
    set_faces_tovao(cvao, cfaces)
    set_uniform_v3f(head_shader, cpts, "controls")
    set_uniform_v3f(head_shader, cpts_w, "weights")
    set_uniform_v3f(head_shader, cpts_canonical, "canonical")
    set_uniform_v3f(control_shader, cpts, "controls")
    set_uniform_v3f(control_shader, cpts_w, "weights")
    set_uniform_v3f(control_shader, get_colors(), "color")
    cpts_isdirty = False

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        my_process_inputs()
        imgui.new_frame()
        # ====================================================
        # imgui_main_window
        # ====================================================
        global show_control_panel, show_metric_window, show_texture_map, \
            head_transparency, control_transparency, control_size, render_head, render_control
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False
                )
                if clicked_quit:
                    break
                imgui.end_menu()
            if imgui.begin_menu("Window", True):
                clicked, show_control_panel = imgui.menu_item(
                    "Show Panel", None, show_control_panel
                )
                clicked, show_metric_window = imgui.menu_item(
                    "Show Metrics", None, show_metric_window
                )
                clicked, show_texture_map = imgui.menu_item(
                    "Show Textures", None, show_texture_map
                )
                imgui.end_menu()

            imgui.end_main_menu_bar()

        if show_control_panel:
            if imgui.begin("Control Panel", True):
                # metric window
                io = imgui.get_io()
                imgui.text(f"fps = {io.framerate:.2f} frame / second")
                imgui.text(f"vertice # = {len(verts)}, face # = {len(faces)}")
                imgui.text(f"control point # = {len(cpts)}")
                imgui.text('')
                imgui.text("Visualization")
                changed, render_head = imgui.checkbox(
                    "show head", render_head,
                )
                changed, head_transparency = imgui.slider_float(
                    "head alpha", head_transparency, 0, 1, "alpha = %.3f"
                )
                changed, render_control = imgui.checkbox(
                    "show points", render_control,
                )
                changed, control_transparency = imgui.slider_float(
                    "points alpha", control_transparency, 0, 1, "alpha = %.3f"
                )
                changed, control_size = imgui.slider_float(
                    "points size", control_size, 0.2, 5, "alpha = %.3f"
                )

                # TODO: reset and redo the grag
                # imgui.text('')
                # imgui.text("Controls")
                # if imgui.button(label="reset control"):
                #     new_texture_file = filedialog.askopenfilename(
                #         title="open texture",
                #         initialdir=os.path.dirname(texture_file_path),
                #         filetypes=[('image file', '.png .jpg .jpeg')])
                #     if len(new_texture_file) > 0:
                #         new_texture_file = os.path.abspath(new_texture_file)
                #         texture = imageio.imread(new_texture_file)[..., :3]
                #         set_texture2d(texture, texture_id)
                #         texture_file_path = new_texture_file

                imgui.text('')
                imgui.text("Contents")
                changed, _ = imgui.input_text(
                    label="", value=texture_file_path, buffer_length=400
                )
                imgui.same_line()
                if imgui.button(label="open texture"):
                    new_texture_file = filedialog.askopenfilename(
                        title="open texture",
                        initialdir=os.path.dirname(texture_file_path),
                        filetypes=[('image file', '.png .jpg .jpeg')])
                    if len(new_texture_file) > 0:
                        new_texture_file = os.path.abspath(new_texture_file)
                        texture = imageio.imread(new_texture_file)[..., :3]
                        set_texture2d(texture, texture_id)
                        texture_file_path = new_texture_file

                # changed, _ = imgui.input_text(
                #     label="", value=obj_file_path, buffer_length=400
                # )
                # imgui.same_line()
                # if imgui.button(label="open mesh"):
                #     new_obj_file = filedialog.askopenfilename(
                #         title="open mesh",
                #         initialdir=os.path.dirname(obj_file_path),
                #         filetypes=[('mesh file', '.obj')])
                #     if len(new_obj_file) > 0:
                #         new_obj_file = os.path.abspath(new_obj_file)
                #         mesh = trimesh.load(new_obj_file)
                #         # cpts_file = np.load(os.path.dirname(new_obj_file) + "/cpoint.npz")
                #         verts = mesh.vertices.astype(np.float32)
                #         verts = np.concatenate([verts, np.ones_like(verts[:, :1])], -1)
                #         faces = mesh.faces.astype(np.int32)
                #         uvs = mesh.visual.uv.astype(np.float32)
                #         uvs[:, 1] = (1 - uvs[:, 1])
                #         set_attribute(head_shader, ['position', 'uv'], [verts, uvs], vao, bufferids)
                #         set_faces_tovao(vao, faces)
                #         obj_file_path = new_obj_file

                changed, _ = imgui.input_text(
                    label="", value=cpts_file_path, buffer_length=400
                )
                imgui.same_line()
                if imgui.button(label="open controls"):
                    new_cpts_file = filedialog.askopenfilename(
                        title="open controls",
                        initialdir=os.path.dirname(cpts_file_path),
                        filetypes=[('npz', '.npz')])
                    if len(new_cpts_file) > 0:
                        new_cpts_file = os.path.abspath(new_cpts_file)
                        new_cpts = np.load(new_cpts_file)
                        print("Will only load cpts")
                        new_cpts = new_cpts["cpts"].astype(np.float32)
                        if new_cpts.shape == (96, 3):  # is an edit file
                            cpts = new_cpts
                            set_uniform_v3f(head_shader, cpts, "controls")
                            set_uniform_v3f(control_shader, cpts, "controls")
                            cpts_isdirty = False
                            cpts_file_path = new_cpts_file
                        elif new_cpts.shape[-2:] == (96, 3) and len(new_cpts.shape) == 3:
                            cpts = new_cpts[new_cpts["frameidx"]]
                            set_uniform_v3f(head_shader, cpts, "controls")
                            set_uniform_v3f(control_shader, cpts, "controls")
                            cpts_isdirty = False
                            cpts_file_path = new_cpts_file

                global render_success, render_success_time
                imgui.text('')
                imgui.text("Render")
                if imgui.button(label="Export Settings"):
                    pose, intrin = get_camera_pose_for_nerf()
                    savedir = os.path.join(os.path.dirname(obj_file_path), "cpoint_edit_0.npz")
                    sufix = 1
                    while os.path.exists(savedir):
                        savedir = os.path.join(os.path.dirname(obj_file_path), f"cpoint_edit_{sufix}.npz")
                        sufix += 1

                    imsavedir = os.path.join(os.path.dirname(obj_file_path), "texture_edit_0.png")
                    sufix = 1
                    while os.path.exists(savedir):
                        savedir = os.path.join(os.path.dirname(obj_file_path), f"texture_edit_{sufix}.npz")
                        sufix += 1

                    np.savez(savedir,
                             frameidx=frame_idx,
                             pose=pose,
                             intrin=intrin,
                             cpts=cpts
                             )
                    imageio.imwrite(imsavedir,
                                    texture)
                    render_success = True
                    render_success_time = time.time()
                if render_success:
                    imgui.same_line()
                    imgui.text("Success!")
                if time.time() - render_success_time > 1:
                    render_success = False

                if imgui.button(label='Launch Rendering'):
                    pass

                imgui.end()

        if show_texture_map:
            if imgui.begin("Texture Maps", True):
                global pen_size, pen_color, pen
                x1, y1 = imgui.get_window_content_region_min()
                x2, y2 = imgui.get_window_content_region_max()
                ww, wh = x2 - x1, y2 - y1
                sz = min(ww, wh)

                imgui.image_button(texture_id, sz, sz, frame_padding=0)  # clicked
                if imgui.is_item_active():
                    cx, cy = glfw.get_cursor_pos(window)
                    wx, wy = imgui.get_window_position()
                    wx, wy = wx + x1, wy + y1

                    x, y = (cx - wx) / sz, (cy - wy) / sz
                    tx, ty = texture.shape[:2]
                    x, y = int(x * (tx - 1)), int(y * (ty - 1))

                    # draw
                    x = np.clip(x, pen_size, tx - pen_size - 1)
                    y = np.clip(y, pen_size, ty - pen_size - 1)
                    original = texture[y - pen_size: y + pen_size + 1, x - pen_size: x + pen_size + 1]
                    edit = pen[..., :3] * 255 * pen[..., -1:] + original * (1 - pen[..., -1:])
                    edit = np.clip(edit, 0, 255).astype(np.uint8)
                    texture[y - pen_size: y + pen_size + 1, x - pen_size: x + pen_size + 1] = edit

                    update_texture2d(texture, texture_id, (0, 0))

                changed1, pen_size = imgui.slider_int(
                    "", pen_size, 3, 40, "pen sz = %d px"
                )
                changed2, pen_color = imgui.color_edit4(
                    "", *pen_color, show_alpha=True
                )
                if changed1 or changed2 or pen is None:
                    psz = pen_size * 2 + 1
                    pen = np.ones((psz, psz, 4)).astype(np.float32) * pen_color
                    radiusx, radiusy = np.meshgrid(np.arange(-pen_size, pen_size + 1),
                                                   np.arange(-pen_size, pen_size + 1))
                    radius = radiusx ** 2 + radiusy ** 2
                    mask = radius <= pen_size ** 2
                    pen[..., -1] *= mask.astype(np.float32)

                imgui.end()

        if show_metric_window:
            show_metric_window = imgui.show_metrics_window(closable=show_metric_window)

        global window_h, window_w
        window_w, window_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, window_w, window_h)
        glClearColor(*background_color)
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # update the view mat
        view_mat, project_mat = get_view_proj_mat()
        set_uniform_mat4(head_shader, view_mat, "view_matrix")
        set_uniform_mat4(head_shader, project_mat, "projection_matrix")
        set_uniform_1f(head_shader, head_transparency, "transparency")
        set_uniform_mat4(control_shader, view_mat, "view_matrix")
        set_uniform_mat4(control_shader, project_mat, "projection_matrix")
        set_uniform_1f(control_shader, control_transparency, "transparency")
        set_uniform_1f(control_shader, control_size, "size")

        # update control points
        if cpts_isdirty:
            set_uniform_v3(control_shader, cpts[cpts_seleted_idx], f"controls[{cpts_seleted_idx}]")
            set_uniform_v3(head_shader, cpts[cpts_seleted_idx], f"controls[{cpts_seleted_idx}]")
            cpts_isdirty = False

        if render_control:
            glUseProgram(control_shader)
            glBindVertexArray(cvao)
            glDrawElementsInstanced(GL_TRIANGLES, len(cfaces.reshape(-1)), GL_UNSIGNED_INT, None, len(cpts))

        if render_head:
            glUseProgram(head_shader)
            glBindVertexArray(vao)
            glDrawElements(GL_TRIANGLES, len(faces.reshape(-1)), GL_UNSIGNED_INT, None)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == '__main__':
    main()
