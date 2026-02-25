import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from RobotEnv import MobileRobotEnv

if __name__ == "__main__":
    # 1. Ortamı Başlat
    env = MobileRobotEnv(grid_size=10.0, num_obstacles=5)

    # 2. Eğitilmiş Modeli (Checkpoint) Yükle
    print("Eğitilmiş model yükleniyor...")

    model_path = "./robot_checkpoints/ppo_model_1100000_steps"
    model = PPO.load(model_path)

    print("Model başarıyla yüklendi!\n")

    # 3. Test İçin Bölüm (Episode) Sayısı
    episodes = 10

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"--- Test Bölümü {ep + 1} Başlıyor ---")

        while not done:
            # Modelden en iyi aksiyonu iste (deterministic=True)
            action, _states = model.predict(obs, deterministic=True)

            # Aksiyonu uygula
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Görselleştir
            env.render()

        print(f"Bölüm {ep + 1} Bitti! | Adım: {step_count} | Ödül: {total_reward}")

    print("\nTest tamamlandı. Çıkmak için grafik penceresini kapatın.")

    plt.ioff()
    plt.show()
    env.close()