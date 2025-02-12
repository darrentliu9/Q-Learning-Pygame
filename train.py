import pygame
import random
import numpy as np
from collections import defaultdict

# 游戏参数
WIDTH = 400
HEIGHT = 600
FPS = 30

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)  

# 初始化Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("QLearning Game")

class GameEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.player = pygame.Rect(WIDTH//2-15, HEIGHT-60, 30, 30)
        self.enemies = []
        self.bullets = []
        self.score = 0
        self.steps = 0
        return self.get_state()
    
    def get_state(self):
        nearest_enemy = min(self.enemies, key=lambda e: e.y, default=None)
        return (
            self.player.x // 20,  # 修改离散粒度
            1 if nearest_enemy else 0,
            (nearest_enemy.x - self.player.x) // 20 if nearest_enemy else 0,
            (nearest_enemy.y - self.player.y) // 20 if nearest_enemy else 0
        )
    
    def step(self, action):
        reward = 0
        done = False
        
        # 修改移动逻辑
        if action == 0:  # 左移
            self.player.x = max(0, self.player.x - 10)
        elif action == 1:  # 右移
            self.player.x = min(WIDTH - 30, self.player.x + 10)
        elif action == 2:  # 修改子弹尺寸
            self.bullets.append(pygame.Rect(
                self.player.centerx-5, self.player.top, 10, 20))
        
        self.steps += 1
        
        # 调整敌机生成频率
        if random.random() < 0.03:  
            self.enemies.append(pygame.Rect(
                random.randint(0, WIDTH-30), -30, 30, 30))
        
        # 修改子弹速度
        for bullet in self.bullets[:]:
            bullet.y -= 8  
            if bullet.y < 0:
                self.bullets.remove(bullet)
        
        # 优化碰撞检测
        player_hitbox = pygame.Rect(
            self.player.x+5, self.player.y+5, 
            self.player.width-10, self.player.height-10)
        
        for enemy in self.enemies[:]:
            enemy.y += 4  
            if player_hitbox.colliderect(enemy):
                done = True
                reward -= 50  # 调整惩罚值
            # 优化子弹碰撞检测
            bullet_collided = False
            for bullet in self.bullets[:]:
                if bullet.colliderect(enemy):
                    self.score += 20  # 提高得分
                    reward += 30      # 提高奖励
                    self.enemies.remove(enemy)
                    self.bullets.remove(bullet)
                    bullet_collided = True
                    break
            if bullet_collided:
                continue
            if enemy.y > HEIGHT:
                self.enemies.remove(enemy)
                reward -= 3  # 降低惩罚
        
        reward += 2  # 提高生存奖励
        
        if self.steps > 2000 or done:  # 延长游戏时间
            done = True
        
        return self.get_state(), reward, done

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        # 初始化随机Q值
        self.q_table = defaultdict(lambda: [random.uniform(-0.1,0.1) for _ in range(3)])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def save(self, file_name):
        np.save(file_name, dict(self.q_table))

    def load(self, file_name):
        self.q_table = defaultdict(lambda: [0, 0, 0], np.load(file_name, allow_pickle=True).item())

def train():
    env = GameEnv()
    agent = QLearningAgent(alpha=0.2, epsilon=0.2)  # 提高学习率和探索率
    
    episodes = 5000  # 增加训练轮次
    total_score = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            
            if done:
                total_score += env.score
                break
        
        if (episode+1) % 100 == 0:
            print(f"Episode: {episode+1}, Avg Score: {total_score/100}")
            total_score = 0
        
    agent.save("q_table.npy")

def test():
    env = GameEnv()
    agent = QLearningAgent(epsilon=0)
    agent.load("q_table.npy")  # 确保加载训练结果
    
    while True:
        state = env.reset()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            state = next_state
            
            # 增强画面渲染
            screen.fill(BLACK)
            
            # 绘制玩家飞机
            pygame.draw.rect(screen, BLUE, env.player)
            pygame.draw.rect(screen, WHITE, env.player, 2)
            
            # 绘制敌机
            for enemy in env.enemies:
                pygame.draw.rect(screen, RED, enemy)
                pygame.draw.line(screen, (200,0,0), 
                              (enemy.left, enemy.top),
                              (enemy.right, enemy.bottom), 3)
            
            # 绘制子弹（黄色）
            for bullet in env.bullets:
                pygame.draw.rect(screen, YELLOW, bullet)
                pygame.draw.circle(screen, (255,200,0), 
                                 (bullet.centerx, bullet.centery), 3)
            
            # 增强得分显示
            font = pygame.font.SysFont("Arial", 24, bold=True)
            score_text = font.render(f"Score: {env.score}", True, WHITE)
            screen.blit(score_text, (20, 20))
            
            pygame.display.flip()
            clock.tick(FPS)
            
            if done:
                break

if __name__ == "__main__":
    train()
    test()