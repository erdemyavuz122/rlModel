import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from RobotEnv import MobileRobotEnv
from classic_planners import AStarPlanner, RRTPlanner
import time

if __name__ == "__main__":
    # 1. Ortamı Başlat ve Başlangıç Durumunu Kaydet
    env = MobileRobotEnv(grid_size=10.0, num_obstacles=5)
    obs, _ = env.reset()

    start_pos = env.robot_pos.copy()
    start_theta = env.robot_theta
    goal_pos = env.goal_pos.copy()
    obstacles = env.obstacles.copy()

    print("Harita oluşturuldu. Algoritmalar hesaplanıyor...\n")

    # --- 2. KLASİK YÖNTEMLERİ HESAPLA ---
    a_star = AStarPlanner(grid_size=10.0, obstacles=obstacles, obstacle_radius=0.6)
    path_astar = a_star.plan(start_pos, goal_pos)

    rrt = RRTPlanner(grid_size=10.0, obstacles=obstacles, obstacle_radius=0.6)
    path_rrt = rrt.plan(start_pos, goal_pos)


    # --- 3. CANLI ANİMASYON FONKSİYONU ---
    def animate_path(path, title_text, color_code):
        if not path:
            print(f"{title_text} için yol bulunamadı!")
            return

        print(f"Sahnede: {title_text}")
        for i in range(len(path)):
            env.robot_pos = np.array(path[i])

            # Görsellik için robotun burnunu (theta) gideceği noktaya çevir
            if i < len(path) - 1:
                dy = path[i + 1][1] - env.robot_pos[1]
                dx = path[i + 1][0] - env.robot_pos[0]
                env.robot_theta = np.arctan2(dy, dx)

            env.render()
            # Başlığı ekle
            env.ax.set_title(title_text, fontsize=14, fontweight='bold', color=color_code)
            plt.pause(0.05)

        time.sleep(1.5)  # Hedefe varınca 1.5 saniye bekle


    # --- 4. SAHNE 1: A* ANİMASYONU ---
    animate_path(path_astar, "1. Klasik Algoritma: A* (A-Star)", "green")

    # --- 5. SAHNE 2: RRT ANİMASYONU ---
    # Robotu başlangıç noktasına geri ışınla
    env.robot_pos = start_pos.copy()
    animate_path(path_rrt, "2. Klasik Algoritma: RRT (Rastgele Ağaç)", "red")

    # --- 6. SAHNE 3: YAPAY ZEKA (PPO) CANLI SÜRÜŞ ---
    print("Sahnede: Yapay Zeka (PPO)")
    # LÜTFEN BURAYA KENDİ 1.2 MİLYON ADIMLIK DOSYANIZIN ADINI YAZIN:
    model_path = "./robot_checkpoints/ppo_model_1200000_steps"
    model = PPO.load(model_path)

    # Ortamı tamamen ilk saniyesine (sıfıra) döndür
    env.robot_pos = start_pos.copy()
    env.robot_theta = start_theta
    obs = env._get_obs()

    done = False
    step_count = 0
    while not done and step_count < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        env.render()
        env.ax.set_title("3. Yapay Zeka Ajanı: PPO (1.2 Milyon Adım Eğitim)", fontsize=14, fontweight='bold',
                         color="green")
        plt.pause(0.05)
        step_count += 1

    print("\nTüm animasyonlar tamamlandı!")
    time.sleep(2)
    plt.ioff()
    plt.show()