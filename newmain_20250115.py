# 必要なモジュールをインポート
import glfw  # OpenGLのウィンドウ操作ライブラリ
import OpenGL.GL as gl  # OpenGLの機能をPythonで利用するためのモジュール
from imgui.integrations.glfw import GlfwRenderer  # ImGuiとGLFWの統合モジュール
import imgui  # グラフィカルUIライブラリ
import numpy as np  # 数値計算用ライブラリ
import util  # カスタムユーティリティモジュール（ユーザー定義）
import util_gau
import imageio  # 画像の入出力ライブラリ
import util_gau  # ガウスデータ関連のカスタムモジュール
import tkinter as tk  # ファイル選択ダイアログ用GUIライブラリ
from tkinter import filedialog  # tkinterのファイルダイアログモジュール
import os  # ファイル操作用モジュール
import sys  # システム関連のモジュール
import argparse  # コマンドライン引数処理用モジュール
from renderer_ogl import OpenGLRenderer, GaussianRenderBase  # レンダリング関連クラスのインポート
#マウスとキーボードを操作するためのやつ
import pyautogui


# 現在のスクリプトのパスを取得して、Pythonのパスに追加
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# スクリプトのディレクトリを作業ディレクトリに設定
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# カメラの初期化
g_camera = util.Camera(1080, 1920)  #高さ1080幅1920pxのカメラ
BACKEND_OGL = 0  # OpenGLバックエンド
BACKEND_CUDA = 1  # CUDAバックエンド
g_renderer_list = [None]  # レンダラーリストの初期化
g_renderer_idx = BACKEND_OGL  # デフォルトでOpenGLを選択
g_renderer: GaussianRenderBase = g_renderer_list[g_renderer_idx]  # 現在選択されているレンダラー
g_scale_modifier = 1.0  # スケール修正用変数
g_auto_sort = True  # 自動ソートの有効化フラグ
g_show_control_win = True  # コントロールウィンドウを表示するか
g_show_help_win = True  # ヘルプウィンドウを表示するか
g_show_camera_win = False  # カメラウィンドウを表示するか
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", 
                        "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]  # 描画モード
g_render_mode = 7  # デフォルトの描画モード（SH:0~3）

# GLFWウィンドウの初期化関数
def impl_glfw_init():
    window_name = "NeUVF editor"  # ウィンドウ名

    if not glfw.init():  # GLFWを初期化。失敗したら終了
        print("Could not initialize OpenGL context")
        exit(1)

    # OpenGLのバージョンやプロファイルを指定
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # ウィンドウの作成
    global window  # グローバル変数として定義
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)  # 作成したウィンドウを現在のコンテキストに設定
    glfw.swap_interval(0)  # V-Syncを無効化

    if not window:  # ウィンドウ作成失敗時の処理
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window  # ウィンドウを返す

# マウスカーソルの移動時のコールバック関数
def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:  # UIがマウス入力を受け取っている場合はカメラを無効化
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)  # カメラにマウスの位置を渡して処理

# マウスボタンの押下/離上時のコールバック関数
def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:  # UIがマウス操作を受け取っている場合は処理しない
        return
    pressed = action == glfw.PRESS  # マウスボタンが押された場合
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)  # 左ボタン押下状態の更新
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)  # 右ボタン押下状態の更新

# マウスホイールのスクロール時のコールバック関数
def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)  # カメラにスクロール量を渡して処理

# キーボード入力のコールバック関数
def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:  # キーが押されたまたは保持された場合
        if key == glfw.KEY_Q:  # Qキーでカメラを右に回転
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:  # Eキーでカメラを左に回転
            g_camera.process_roll_key(-1)

# カメラの位置情報を遅延更新
def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:  # カメラの位置情報が変更された場合のみ更新
        g_renderer.update_camera_pose(g_camera)  # レンダラーに新しいカメラの位置情報を渡す
        g_camera.is_pose_dirty = False  # 更新完了後、変更フラグをリセット

# カメラの内部パラメータを遅延更新
def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:  # カメラの内部パラメータが変更された場合のみ更新
        g_renderer.update_camera_intrin(g_camera)  # レンダラーに新しい内部パラメータを渡す
        g_camera.is_intrin_dirty = False  # 更新完了後、変更フラグをリセット

# レンダラーの状態を更新
def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)  # ガウスデータをレンダラーに渡す
    g_renderer.sort_and_update(g_camera)  # ソートと更新処理
    g_renderer.set_scale_modifier(g_scale_modifier)  # スケール修正値を設定
    g_renderer.set_render_mod(g_render_mode - 3)  # レンダリングモードを設定
    g_renderer.update_camera_pose(g_camera)  # カメラの位置情報を更新
    g_renderer.update_camera_intrin(g_camera)  # カメラの内部パラメータを更新
    g_renderer.set_render_reso(g_camera.w, g_camera.h)  # レンダリング解像度を更新

# ウィンドウのサイズ変更時のコールバック関数
def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)  # OpenGLのビューポートサイズを更新
    g_camera.update_resolution(height, width)  # カメラの解像度を更新
    g_renderer.set_render_reso(width, height)  # レンダラーの解像度を更新

# メイン関数
def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables
        
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

    # init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
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
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                changed, g_renderer.reduce_updates = imgui.checkbox(
                        "reduce updates", g_renderer.reduce_updates,
                    )

                imgui.text(f"# of Gaus = {len(gaussians)}")
                if imgui.button(label='open ply'):
                    file_path = filedialog.askopenfilename(title="open ply",
                        initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                        filetypes=[('ply file', '.ply')]
                        )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                        except RuntimeError as e:
                            pass
                
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
                changed, g_auto_sort = imgui.checkbox(
                        "auto sort", g_auto_sort,
                    )
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)
                
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
                    # np.savez(
                    #     "save.npz",
                    #     gau_xyz=gaussians.xyz,
                    #     gau_s=gaussians.scale,
                    #     gau_rot=gaussians.rot,
                    #     gau_c=gaussians.sh,
                    #     gau_a=gaussians.opacity,
                    #     viewmat=g_camera.get_view_matrix(),
                    #     projmat=g_camera.get_project_matrix(),
                    #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                    # )
                imgui.end()

        if g_show_camera_win:
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
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()

