from plyfile import PlyData
import numpy as np

def calculate_gaussian_center(ply_file_path):
    """
    PLYファイルからすべてのガウシアンの位置の平均を計算し、重心を求める
    :param ply_file_path: PLYファイルのパス
    :return: ガウシアンの重心 (x, y, z)
    """
    # PLYファイルを読み込む
    ply_data = PlyData.read(ply_file_path)

    # 頂点データを取得
    vertices = ply_data['vertex']

    # x, y, z 座標を取得
    x_coords = vertices['x']
    y_coords = vertices['y']
    z_coords = vertices['z']

    # 平均値を計算 (重心)
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    center_z = np.mean(z_coords)

    return center_x, center_y, center_z

# 使用例
if __name__ == "__main__":
    ply_file_path = r"C:\Users\81803\research\output\oga\oga_point_cloud.ply"  # PLYファイルのパスを指定
    center = calculate_gaussian_center(ply_file_path)
    print(f"Gaussian Center (Centroid): x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}")
