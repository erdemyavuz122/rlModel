import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RobotEnv import MobileRobotEnv
from classic_planners import AStarPlanner, RRTPlanner


def calculate_path_length(path):
    """Bir rotanın toplam uzunluğunu hesaplar"""
    length = 0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        length += np.linalg.norm(p1 - p2)
    return length


if __name__ == "__main__":
    # 1. Ortamı Hazırla (Eğitimdeki parametrelerle aynı)
    env = MobileRobotEnv(grid_size=10.0, num_obstacles=5)
    obs, info = env.reset()  # Rastgele bir harita oluştur

    start_pos = env.robot_pos
    goal_pos = env.goal_pos
    obstacles = env.obstacles

    print("Harita oluşturuldu. Algoritmalar yarışıyor...\n")

    # --- 2. KLASİK YÖNTEM 1: A* (A-STAR) ---
    print("A* Planlanıyor...")
    a_star = AStarPlanner(grid_size=10.0, obstacles=obstacles, obstacle_radius=0.6)
    path_astar = a_star.plan(start_pos, goal_pos)
    len_astar = calculate_path_length(path_astar) if path_astar else 0
    print(f"A* Tamamlandı. Yol Uzunluğu: {len_astar:.2f}")

    # --- 3. KLASİK YÖNTEM 2: RRT ---
    print("RRT Planlanıyor...")
    rrt = RRTPlanner(grid_size=10.0, obstacles=obstacles, obstacle_radius=0.6)
    path_rrt = rrt.plan(start_pos, goal_pos)
    len_rrt = calculate_path_length(path_rrt) if path_rrt else 0
    print(f"RRT Tamamlandı. Yol Uzunluğu: {len_rrt:.2f}")

    # --- 4. YAPAY ZEKA: PPO (RL AGENT) ---
    print("PPO Ajanı Yükleniyor...")
    # Checkpoint yolunuzu buraya yazın (veya final modeli)
    model_path = "./robot_checkpoints/ppo_model_1200000_steps"
    model = PPO.load(model_path)

    path_ppo = [start_pos]
    curr_obs = obs
    done = False
    max_steps = 200

    for _ in range(max_steps):
        action, _ = model.predict(curr_obs, deterministic=True)
        curr_obs, reward, terminated, truncated, _ = env.step(action)
        path_ppo.append(env.robot_pos.copy())  # Robotun yeni konumunu kaydet
        if terminated or truncated:
            break

    len_ppo = calculate_path_length(path_ppo)
    print(f"PPO Tamamlandı. Yol Uzunluğu: {len_ppo:.2f}")

    # --- 5. GÖRSELLEŞTİRME VE KIYASLAMA ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_facecolor('#BDC3C7')  # Zemin rengi

    # Engelleri Çiz
    for o in obstacles:
        circle = plt.Circle((o[0], o[1]), 0.6, color='#34495E', zorder=2)
        ax.add_patch(circle)

    # Hedefi ve Başlangıcı Çiz
    ax.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Hedef', zorder=5)
    ax.plot(start_pos[0], start_pos[1], 'ks', markersize=8, label='Başlangıç', zorder=5)

    # Rotaları Çiz
    if path_astar:
        path_astar = np.array(path_astar)
        ax.plot(path_astar[:, 0], path_astar[:, 1], 'g--', linewidth=2, label=f'A* (Optimum): {len_astar:.2f}m')

    if path_rrt:
        path_rrt = np.array(path_rrt)
        ax.plot(path_rrt[:, 0], path_rrt[:, 1], 'r-.', linewidth=2, label=f'RRT (Keşifçi): {len_rrt:.2f}m')

    path_ppo = np.array(path_ppo)
    ax.plot(path_ppo[:, 0], path_ppo[:, 1], 'b-', linewidth=3, label=f'PPO (Yapay Zeka): {len_ppo:.2f}m')

    ax.legend(loc='upper right')
    ax.set_title("Algoritma Karşılaştırması: RL vs A* vs RRT")
    plt.show()