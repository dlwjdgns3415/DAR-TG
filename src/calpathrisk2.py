import numpy as np
import pickle

def load_lidar_data(file_path, grid_size=0.5, grid_map_size=100):
    """📌 .pkl 파일에서 LiDAR 데이터를 로드하여 2D Grid Map을 생성"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # ✅ (x, y) 좌표만 추출
    lidar_points = np.vstack(data['lidar'])[:, :2]

    # ✅ 2D Grid Map 생성
    lidar_grid_map = np.zeros((grid_map_size, grid_map_size))

    for lidar_pt in lidar_points:
        x_idx = int(lidar_pt[0] / grid_size)
        y_idx = int(lidar_pt[1] / grid_size)
        if 0 <= x_idx < grid_map_size and 0 <= y_idx < grid_map_size:
            lidar_grid_map[x_idx, y_idx] += 1  # 🔥 밀집도 증가

    return lidar_grid_map


def calculate_density(lidar_grid_map, waypoints, grid_size=0.5, grid_map_size=100):
    """📌 각 Waypoint가 속한 Grid의 밀집도 계산"""
    density_values = []

    for waypoint in waypoints:
        x_idx = int(waypoint[0] / grid_size)
        y_idx = int(waypoint[1] / grid_size)

        if 0 <= x_idx < grid_map_size and 0 <= y_idx < grid_map_size:
            density_values.append(lidar_grid_map[x_idx, y_idx])  # 🔥 밀집도 가져오기
        else:
            density_values.append(0)  # 범위를 벗어나면 0

    # ✅ 밀집도 정규화 (Threshold + Min-Max Scaling)
    MIN_DENSITY = 1
    MAX_DENSITY = 500

    density_values = np.array(density_values, dtype=np.float32)
    print(density_values)
    density_values = (density_values - MIN_DENSITY) / (MAX_DENSITY - MIN_DENSITY)
    density_values = np.clip(density_values, 0, 1)  # 🔥 0~1 사이로 제한

    return density_values


def calculate_path_risk_score(density_values):
    """📌 Path Risk Score 계산 (경로 전체의 평균 장애물 밀집도)"""
    N = len(density_values)  # Waypoints 개수
    if N == 0:
        return 0.0  # Waypoints가 없으면 Path Risk Score는 0
    P_risk = np.sum(density_values) / N  # 평균 장애물 밀집도 계산
    return P_risk


if __name__ == "__main__":
    # 📌 1. LiDAR 데이터 로드 및 Grid 변환
    lidar_file = "/home/dke/jhlee/DTG/data_sample/data_folder/0_20.pkl"  # 사용자의 .pkl 파일 경로
    lidar_grid_map = load_lidar_data(lidar_file, grid_size=0.5, grid_map_size=100)

    # 📌 2. Waypoints 로드 (예제 데이터를 사용, 실제 데이터에 맞게 수정)
    waypoints = np.array([
        [1.0, 2.0], [2.5, 3.0], [3.0, 5.0], [4.5, 6.5],
        [5.0, 8.0], [6.5, 9.5], [7.0, 10.0], [8.5, 11.5]
    ])  # 예제 Waypoints

    # 📌 3. Waypoints의 장애물 밀집도 계산
    density_values = calculate_density(lidar_grid_map, waypoints, grid_size=0.5, grid_map_size=100)

    # 📌 4. Path Risk Score 계산
    P_risk = calculate_path_risk_score(density_values)

    # 결과 출력
    print("Waypoints별 장애물 밀집도:", density_values)
    print("Path Risk Score:", P_risk)
