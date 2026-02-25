import numpy as np
import heapq


class AStarPlanner:
    """Sürekli (Continuous) uzayı Grid'e çevirerek çalışan A* Planlayıcı"""

    def __init__(self, grid_size=10.0, resolution=0.2, obstacles=None, obstacle_radius=0.6):
        self.grid_size = grid_size
        self.resolution = resolution  # Haritayı kaçar birimlik karelere böleceğiz (örn: 0.2)
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius

        # Grid boyutlarını hesapla (10.0 / 0.2 = 50x50'lik bir matris)
        self.width = int(grid_size / resolution)
        self.height = int(grid_size / resolution)

    def _get_grid_pos(self, pos):
        """Gerçek X,Y koordinatını Grid indeksine (satır, sütun) çevirir"""
        x = min(int(pos[0] / self.resolution), self.width - 1)
        y = min(int(pos[1] / self.resolution), self.height - 1)
        return (x, y)

    def _get_continuous_pos(self, grid_pos):
        """Grid indeksini tekrar gerçek X,Y koordinatına çevirir"""
        x = grid_pos[0] * self.resolution
        y = grid_pos[1] * self.resolution
        return np.array([x, y])

    def _is_valid(self, grid_pos):
        """Bu karenin içi boş mu (duvar veya engel var mı) kontrol eder"""
        x, y = grid_pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False  # Harita dışı

        cont_pos = self._get_continuous_pos(grid_pos)
        for obs in self.obstacles:
            # Robotun kendi genişliğini de hesaba katarak güvenlik payı bırakıyoruz
            if np.linalg.norm(cont_pos - obs) <= (self.obstacle_radius + 0.3):
                return False  # Engele çok yakın
        return True

    def plan(self, start_pos, goal_pos):
        """A* Algoritmasının ana döngüsü. Başlangıçtan hedefe en kısa yolu bulur."""
        start = self._get_grid_pos(start_pos)
        goal = self._get_grid_pos(goal_pos)

        if not self._is_valid(start) or not self._is_valid(goal):
            return []  # Geçersiz başlangıç veya bitiş noktası

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        # 8 yönlü hareket (Sağ, Sol, Yukarı, Aşağı ve Çaprazlar)
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Hedef bulundu! Rotayı geriye doğru çiz
                path = []
                while current in came_from:
                    path.append(self._get_continuous_pos(current))
                    current = came_from[current]
                path.append(self._get_continuous_pos(start))
                return path[::-1]  # Listeyi ters çevir (Başlangıç -> Hedef)

            for dx, dy in motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self._is_valid(neighbor):
                    continue

                # Çapraz gidişlerin maliyeti (kök 2), düz gidişlerin maliyeti (1)
                cost = np.sqrt(dx ** 2 + dy ** 2) * self.resolution
                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    # Sezgisel (Heuristic) Maliyet: Hedefe olan kuş uçuşu uzaklık
                    h_score = np.linalg.norm(np.array(neighbor) - np.array(goal)) * self.resolution
                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor))

        return []  # Hedefe giden hiçbir yol bulunamadı


import math
import random


class RRTPlanner:
    """Rastgele Dallanarak Yol Bulan (RRT) Planlayıcı"""

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self, grid_size=10.0, obstacles=None, obstacle_radius=0.6, step_size=0.5, max_iter=2000):
        self.grid_size = grid_size
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        self.step_size = step_size  # Her adımda ne kadar dal uzatacak
        self.max_iter = max_iter  # Maksimum deneme sayısı

    def plan(self, start_pos, goal_pos):
        start_node = self.Node(start_pos[0], start_pos[1])
        goal_node = self.Node(goal_pos[0], goal_pos[1])
        self.node_list = [start_node]

        for _ in range(self.max_iter):
            # 1. Rastgele bir nokta seç (Bazen doğrudan hedefi seç ki oraya yönelsin)
            if random.randint(0, 100) > 10:
                rnd_node = self.Node(random.uniform(0, self.grid_size), random.uniform(0, self.grid_size))
            else:
                rnd_node = goal_node

            # 2. Ağaçtaki en yakın düğümü bul
            nearest_node = self._get_nearest_node_index(self.node_list, rnd_node)

            # 3. O yöne doğru bir dal uzat (New Node)
            theta = math.atan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
            new_node = self.Node(nearest_node.x + self.step_size * math.cos(theta),
                                 nearest_node.y + self.step_size * math.sin(theta))
            new_node.parent = nearest_node

            # 4. Eğer engellere çarpmıyorsa ağaca ekle
            if not self._check_collision(new_node, self.obstacles):
                self.node_list.append(new_node)


                dx = new_node.x - goal_node.x
                dy = new_node.y - goal_node.y
                if math.sqrt(dx ** 2 + dy ** 2) <= self.step_size:
                    goal_node.parent = new_node
                    return self._generate_final_course(goal_node)

        return []  # Başarısız olursa boş liste dön

    def _get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    def _check_collision(self, node, obstacles):

        for obs in obstacles:
            dx = node.x - obs[0]
            dy = node.y - obs[1]
            if math.sqrt(dx ** 2 + dy ** 2) <= (self.obstacle_radius + 0.3):
                return True  # Çarpışma var

        # Harita dışına çıktı mı?
        if node.x < 0 or node.x > self.grid_size or node.y < 0 or node.y > self.grid_size:
            return True

        return False

    def _generate_final_course(self, goal_node):
        path = [[goal_node.x, goal_node.y]]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append([node.x, node.y])
        return path[::-1]  # Rotayı ters çevir