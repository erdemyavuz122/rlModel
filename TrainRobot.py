import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from RobotEnv import MobileRobotEnv
import os

if __name__ == "__main__":
    # 1. Kayıt (Checkpoint) ve Log Klasörlerini Oluştur
    checkpoint_dir = "./robot_checkpoints/"
    log_dir = "./robot_tensorboard/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Eğitim Ortamını Başlat
    env = MobileRobotEnv(grid_size=10.0, num_obstacles=5)

    # 3. Checkpoint (Yedekleme) Ayarları
    # Her 100.000 adımda bir modeli 'robot_checkpoints' klasörüne kaydeder.
    # Bilgisayar kapansa bile en son kaydedilen adımdan test yapabilirsiniz.
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="ppo_model"
    )

    # 4. PPO Modelini Tanımla (TensorBoard aktif edildi)
    print("Model oluşturuluyor...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log=log_dir)

    # 5. Tüm Gün Sürecek Eğitimi Başlat

    print("Tüm gün sürecek eğitim başlıyor! PyCharm'ı açık bırakın...")
    training_steps = 10000000

    model.learn(
        total_timesteps=training_steps,
        callback=checkpoint_callback,
        tb_log_name="PPO_Training_Day1"
    )

    # 6. Eğitim başarıyla biterse final modelini kaydet
    model.save("ppo_mobile_robot_final")
    print("\nEğitim tamamen bitti ve 'ppo_mobile_robot_final.zip' olarak kaydedildi.")

    env.close()