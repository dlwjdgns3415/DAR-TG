import numpy as np
import pickle
from scipy.spatial import cKDTree  # 빠른 거리 계산을 위한 KDTree 사용

def load_lidar_data(file_path):
    """📌 .pkl 파일에서 LiDAR 데이터를 로드하여 (x, y) 좌표를 반환"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # (x, y) 좌표만 추출
    lidar_points = np.vstack(data['lidar'])[:, :2]  
    return lidar_points

def calculate_density(lidar_points, waypoints, radius=1.0):
    """📌 각 Waypoint 주변 반경 `r` 내 LiDAR 포인트 개수(밀집도) 계산"""
    tree = cKDTree(lidar_points)  # LiDAR 데이터를 KDTree로 변환하여 빠른 거리 계산

    density_values = []
    for waypoint in waypoints:
        num_neighbors = len(tree.query_ball_point(waypoint, radius))  # 반경 내 포인트 개수 계산
        density_values.append(num_neighbors)

    # 밀집도 정규화 (최대값 기준으로 0~1 스케일링)
    max_density = 5000
    min_density = 100
    density_values = np.array(density_values, dtype=np.float32)
    print(density_values)
    # ✅ 1. 너무 낮은 값(MIN_DENSITY 이하)은 Min-Max Scaling 적용
    density_values = (density_values - min_density) / (max_density - min_density)
    
    # ✅ 2. 밀집도가 MAX_DENSITY보다 크면 1로 고정 (Threshold 방식)
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
    # 📌 1. LiDAR 데이터 로드
    lidar_file = "/home/dke/jhlee/DTG/data_sample/data_folder/0_88.pkl"  # 사용자의 .pkl 파일 경로
    lidar_points = load_lidar_data(lidar_file)

    # 📌 2. Waypoints 로드 (예제 데이터를 사용, 실제 데이터에 맞게 수정)
    waypoints = np.array([
        [1.0, 2.0], [2.5, 3.0], [3.0, 5.0], [4.5, 6.5],
        [5.0, 8.0], [6.5, 9.5], [7.0, 10.0], [8.5, 11.5]
    ])  # 예제 Waypoints

    # 📌 3. Waypoints의 장애물 밀집도 계산
    density_values = calculate_density(lidar_points, waypoints, radius=1.0)

    # 📌 4. Path Risk Score 계산
    P_risk = calculate_path_risk_score(density_values)

    # 결과 출력
    print("Waypoints별 장애물 밀집도:", density_values)
    print("Path Risk Score:", P_risk)
