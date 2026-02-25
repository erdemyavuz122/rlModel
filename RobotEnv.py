import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class MobileRobotEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size=10.0, num_obstacles=5):
        super(MobileRobotEnv, self).__init__()

        self.grid_size = grid_size
        self.max_steps = 200
        self.current_step = 0
        self.goal_radius = 0.5

        self.num_obstacles = num_obstacles
        self.obstacle_radius = 0.6
        self.obstacles = []

        self.num_sensors = 24
        self.sensor_max_range = 3.0

        self.action_space = spaces.Discrete(3)

        low = np.array([0.0, 0.0, -np.pi, 0.0, 0.0] + [0.0] * self.num_sensors, dtype=np.float32)
        high = np.array([self.grid_size, self.grid_size, np.pi, self.grid_size, self.grid_size] + [
            self.sensor_max_range] * self.num_sensors, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.fig, self.ax = None, None
        self.prev_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.robot_pos = np.random.uniform(1.0, self.grid_size - 1.0, size=(2,))
        self.robot_theta = np.random.uniform(-np.pi, np.pi)

        self.goal_pos = np.random.uniform(1.0, self.grid_size - 1.0, size=(2,))
        while np.linalg.norm(self.robot_pos - self.goal_pos) < 4.0:
            self.goal_pos = np.random.uniform(1.0, self.grid_size - 1.0, size=(2,))

        min_x = min(self.robot_pos[0], self.goal_pos[0])
        max_x = max(self.robot_pos[0], self.goal_pos[0])
        min_y = min(self.robot_pos[1], self.goal_pos[1])
        max_y = max(self.robot_pos[1], self.goal_pos[1])

        padding = 1.5
        spawn_min_x = max(1.0, min_x - padding)
        spawn_max_x = min(self.grid_size - 1.0, max_x + padding)
        spawn_min_y = max(1.0, min_y - padding)
        spawn_max_y = min(self.grid_size - 1.0, max_y + padding)

        self.obstacles = []
        for _ in range(self.num_obstacles):
            attempts = 0
            while attempts < 100:
                obs_x = np.random.uniform(spawn_min_x, spawn_max_x)
                obs_y = np.random.uniform(spawn_min_y, spawn_max_y)
                obs_pos = np.array([obs_x, obs_y])

                dist_to_robot = np.linalg.norm(obs_pos - self.robot_pos)
                dist_to_goal = np.linalg.norm(obs_pos - self.goal_pos)

                # YENİ: Engellerin birbirinin üstüne binmesini engelle
                overlap = False
                for existing_obs in self.obstacles:
                    # İki engelin merkezleri arasındaki mesafe, çaplarından büyük olmalı (ufak bir pay ile)
                    if np.linalg.norm(obs_pos - existing_obs) < (self.obstacle_radius * 2 + 0.1):
                        overlap = True
                        break

                if not overlap and dist_to_robot > 1.5 and dist_to_goal > 1.5:
                    self.obstacles.append(obs_pos)
                    break
                attempts += 1

        self.prev_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        return self._get_obs(), {}

    def _get_sensor_readings(self):
        readings = []
        angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)

        for angle in angles:
            ray_angle = self.robot_theta + angle
            min_dist = self.sensor_max_range

            for step in np.linspace(0, self.sensor_max_range, 20):
                rx = self.robot_pos[0] + step * np.cos(ray_angle)
                ry = self.robot_pos[1] + step * np.sin(ray_angle)

                if rx < 0 or rx > self.grid_size or ry < 0 or ry > self.grid_size:
                    min_dist = min(min_dist, step)
                    break

                hit_obstacle = False
                for obs in self.obstacles:
                    if np.linalg.norm([rx - obs[0], ry - obs[1]]) <= self.obstacle_radius:
                        min_dist = min(min_dist, step)
                        hit_obstacle = True
                        break

                if hit_obstacle:
                    break

            readings.append(min_dist)
        return np.array(readings, dtype=np.float32)

    def _get_obs(self):
        sensors = self._get_sensor_readings()
        obs = [self.robot_pos[0], self.robot_pos[1], self.robot_theta, self.goal_pos[0], self.goal_pos[1]]
        obs.extend(sensors)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        step_size = 0.4
        turn_angle = np.pi / 4

        if action == 0:
            self.robot_pos[0] += step_size * np.cos(self.robot_theta)
            self.robot_pos[1] += step_size * np.sin(self.robot_theta)
        elif action == 1:
            self.robot_theta += turn_angle
        elif action == 2:
            self.robot_theta -= turn_angle

        self.robot_theta = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi
        self.robot_pos = np.clip(self.robot_pos, 0.0, self.grid_size)

        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        is_collision = False
        if self.robot_pos[0] <= 0 or self.robot_pos[0] >= self.grid_size or self.robot_pos[1] <= 0 or self.robot_pos[
            1] >= self.grid_size:
            is_collision = True
        for obs in self.obstacles:
            if np.linalg.norm(self.robot_pos - obs) <= self.obstacle_radius:
                is_collision = True
                break

        progress_reward = (self.prev_distance - distance_to_goal) * 10.0
        self.prev_distance = distance_to_goal

        reward = progress_reward - 0.2

        terminated = False
        truncated = False

        if is_collision:
            reward = -100.0
            terminated = True
        elif distance_to_goal <= self.goal_radius:
            reward = 100.0
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            # Daha şık bir figür oranı ve zemin rengi
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig.patch.set_facecolor('#2C3E50')  # Çerçeve rengi (Koyu)

        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)

        # YENİ: Koordinat sistemini gizle, daha gerçekçi bir zemin yap
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor('#BDC3C7')  # Zemin rengi (Beton grisi)

        # Hedef Bölgesi (Açık yeşil, yarı saydam bir iniş/varış bölgesi)
        goal_zone = plt.Circle((self.goal_pos[0], self.goal_pos[1]), self.goal_radius, color='#2ECC71', alpha=0.6,
                               zorder=2)
        self.ax.add_patch(goal_zone)
        # Hedefin tam ortasına küçük bir işaret (X)
        self.ax.plot(self.goal_pos[0], self.goal_pos[1], marker='X', color='white', markersize=8, zorder=3)

        # Engeller (Siyah/Koyu gri kalın sütunlar)
        for obs in self.obstacles:
            pillar = plt.Circle((obs[0], obs[1]), self.obstacle_radius, color='#34495E', ec='black', lw=2, zorder=3)
            self.ax.add_patch(pillar)

        # Sensör Işınları (Lazer gibi açık mavi)
        sensors = self._get_sensor_readings()
        angles = np.linspace(-np.pi, np.pi, self.num_sensors, endpoint=False)
        for i, angle in enumerate(angles):
            ray_angle = self.robot_theta + angle
            end_x = self.robot_pos[0] + sensors[i] * np.cos(ray_angle)
            end_y = self.robot_pos[1] + sensors[i] * np.sin(ray_angle)
            self.ax.plot([self.robot_pos[0], end_x], [self.robot_pos[1], end_y], color='#3498DB', linestyle='-',
                         linewidth=1.5, alpha=0.4, zorder=1)

        # Robot (Turuncu gövde ve belirgin yön çizgisi)
        robot_body = plt.Circle((self.robot_pos[0], self.robot_pos[1]), 0.3, color='#E67E22', ec='black', lw=1.5,
                                zorder=4)
        self.ax.add_patch(robot_body)
        dx = np.cos(self.robot_theta) * 0.4
        dy = np.sin(self.robot_theta) * 0.4
        self.ax.arrow(self.robot_pos[0], self.robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='black',
                      ec='black', zorder=5)

        plt.pause(0.05)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)