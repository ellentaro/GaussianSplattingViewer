# 必要なモジュールをインポート
import glfw  # OpenGLのウィンドウ操作ライブラリ
import OpenGL.GL as gl  # OpenGLの機能をPythonで利用するためのモジュール
from imgui.integrations.glfw import GlfwRenderer  # ImGuiとGLFWの統合モジュール
import imgui  # グラフィカルUIライブラリ
import numpy as np  # 数値計算用ライブラリ
import util  # カスタムユーティリティモジュール（ユーザー定義）
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

    # Init renderer
    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        g_renderer_idx = BACKEND_OGL
    else:
        g_renderer_idx = BACKEND_CUDA

    g_renderer = g_renderer_list[g_renderer_idx]

    # Load Gaussian data
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)

    # Settings
    frame_count = 0
    image_output_dir = "output_images"
    os.makedirs(image_output_dir, exist_ok=True)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Update camera pose and intrinsics lazily
        update_camera_pose_lazy()
        update_camera_intrin_lazy()

        # Render the scene
        g_renderer.draw()

        # Save the rendered image
        if frame_count % 10 == 0:  # Save every 10th frame
            width, height = glfw.get_framebuffer_size(window)
            bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
            image_path = os.path.join(image_output_dir, f"frame_{frame_count:04d}.png")
            imageio.imwrite(image_path, img[::-1])
            print(f"Saved: {image_path}")

        # Render the UI (optional)
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

        frame_count += 1

    impl.shutdown()
    glfw.terminate()


# エントリーポイント
if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")  # コマンドライン引数の設定
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")  # HiDPIスケーリングオプション
    args = parser.parse_args()  # 引数をパース

    main()  # メイン関数を実行

