import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio


camera = util.Camera(500, 500)

def impl_glfw_init():
    width, height = 500, 500
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

def cursor_pos_callback(window, xpos, ypos):
    camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS)

def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)

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
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1.,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    num_gau = len(gau_xyz)

    # Load and compile shaders
    program = util.load_shaders('shaders/gau_vert.glsl', 'shaders/gau_frag.glsl')

    # load quad geometry
    vao, buffer_id = util.set_attributes(program, ["position"], [quad_v])
    util.set_faces_tovao(vao, quad_f)
    
    # load gaussian geometry
    # util.set_attribute_instanced(program, "g_pos", gau_xyz, vao=vao)
    # util.set_attribute_instanced(program, "g_rot", gau_rot, vao=vao)
    # util.set_attribute_instanced(program, "g_scale", gau_s, vao=vao)
    # util.set_attribute_instanced(program, "g_dc_color", gau_c, vao=vao)
    # util.set_attribute_instanced(program, "g_opacity", gau_a, vao=vao)
    gaussian_data = np.concatenate([
        gau_xyz, gau_rot, gau_s, gau_a, gau_c
    ], axis=-1)
    gaussian_data = np.ascontiguousarray(gaussian_data)
    util.set_storage_buffer_data(program, "gaussian_data", gaussian_data, vao=vao)
    util.set_uniform_1int(program, 3, "sh_dim")
    util.set_uniform_1f(program, 1., "scale_modifier")
    
    view_mat = camera.get_view_matrix()
    proj_mat = camera.get_project_matrix()
    util.set_uniform_mat4(program, proj_mat, "projection_matrix")
    util.set_uniform_mat4(program, view_mat, "view_matrix")
    util.set_uniform_v3(program, camera.get_htanfovxy_focal(), "hfovxy_focal")
    
    # Create position texture for instancing
    # instance_positions = np.random.rand(1000, 3) * 2.0 - 1.0  # Random positions in [-1, 1]
    # texture = gl.glGenTextures(1)
    # gl.glBindTexture(gl.GL_TEXTURE_1D, texture)
    # gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGB, len(instance_positions), 0, gl.GL_RGB, gl.GL_FLOAT, instance_positions)
    # gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    # gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    
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

        view_mat = camera.get_view_matrix()
        util.set_uniform_mat4(program, view_mat, "view_matrix")

        gl.glUseProgram(program)
        gl.glBindVertexArray(vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, len(quad_f.reshape(-1)), gl.GL_UNSIGNED_INT, None, num_gau)

        # imgui ui
        if imgui.begin("Control", True):
            if imgui.button(label='save image'):
                width, height = glfw.get_framebuffer_size(window)
                nrChannels = 3;
                stride = nrChannels * width;
                stride += (4 - stride % 4) if stride % 4 else 0
                bufferSize = stride * height;
                gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                gl.glReadBuffer(gl.GL_FRONT)
                bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                imageio.imwrite("save.png", img[::-1])
                
                # save intermediate information
                np.savez(
                    "save.npz",
                    gau_xyz=gau_xyz,
                    gau_s=gau_s,
                    gau_rot=gau_rot,
                    gau_c=gau_c,
                    gau_a=gau_a,
                    viewmat=camera.get_view_matrix(),
                    projmat=camera.get_project_matrix(),
                    hfovxyfocal=camera.get_htanfovxy_focal()
                )
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()
