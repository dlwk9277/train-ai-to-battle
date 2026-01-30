import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import os

# ANSI color codes for terminal
class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class BattleArena:
    """10x10 grid battle arena with items and combat"""
    
    def __init__(self, size=10):
        self.size = size
        self.reset()
        
    def reset(self):
        self.agent1_pos = [0, 0]
        self.agent2_pos = [self.size-1, self.size-1]
        self.agent1_hp = 100
        self.agent2_hp = 100
        self.agent1_item = None
        self.agent2_item = None
        self.agent1_cooldown = 0
        self.agent2_cooldown = 0
        self.items = {}
        self.spawn_items(3)
        self.steps = 0
        self.max_steps = 300
        self.action_log = deque(maxlen=5)
        return self.get_state(1), self.get_state(2)
    
    def spawn_items(self, count):
        import random
        item_types = ['health', 'attack', 'speed', 'shield']
        for _ in range(count):
            for _ in range(50):  # Try 50 times
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if (x, y) not in self.items and \
                   [x, y] != self.agent1_pos and \
                   [x, y] != self.agent2_pos:
                    self.items[(x, y)] = random.choice(item_types)
                    break
    
    def get_state(self, agent_id):
        state = np.zeros((self.size, self.size, 6))
        if agent_id == 1:
            state[self.agent1_pos[0], self.agent1_pos[1], 0] = 1
            state[self.agent2_pos[0], self.agent2_pos[1], 1] = 1
        else:
            state[self.agent2_pos[0], self.agent2_pos[1], 0] = 1
            state[self.agent1_pos[0], self.agent1_pos[1], 1] = 1
        
        for (x, y), item in self.items.items():
            if item == 'health':
                state[x, y, 2] = 1
            elif item == 'attack':
                state[x, y, 3] = 1
            elif item == 'speed':
                state[x, y, 4] = 1
            elif item == 'shield':
                state[x, y, 5] = 1
        
        return state
    
    def step(self, action1, action2):
        import random
        self.steps += 1
        rewards = [0, 0]
        
        self.agent1_cooldown = max(0, self.agent1_cooldown - 1)
        self.agent2_cooldown = max(0, self.agent2_cooldown - 1)
        
        action_names = ['Up↑', 'Down↓', 'Left←', 'Right→', 'Attack⚔️', 'Use🎒', 
                       'Dodge🏃', 'Teleport⚡', 'Counter🛡️', 'Berserk💥']
        
        self.action_log.append(f"{Colors.BLUE}A1:{Colors.RESET}{action_names[action1]:12s} {Colors.RED}A2:{Colors.RESET}{action_names[action2]:12s}")
        
        self._process_action(1, action1, rewards)
        self._process_action(2, action2, rewards)
        self._check_item_pickup(1, rewards)
        self._check_item_pickup(2, rewards)
        
        if self.steps % 20 == 0 and len(self.items) < 5:
            self.spawn_items(1)
        
        rewards[0] -= 0.1
        rewards[1] -= 0.1
        
        done = False
        winner = None
        if self.agent1_hp <= 0:
            rewards[0] -= 200
            rewards[1] += 200
            done = True
            winner = 2
        elif self.agent2_hp <= 0:
            rewards[0] += 200
            rewards[1] -= 200
            done = True
            winner = 1
        elif self.steps >= self.max_steps:
            if self.agent1_hp > self.agent2_hp:
                rewards[0] += 100
                rewards[1] -= 100
                winner = 1
            elif self.agent2_hp > self.agent1_hp:
                rewards[0] -= 100
                rewards[1] += 100
                winner = 2
            done = True
        
        return self.get_state(1), self.get_state(2), rewards[0], rewards[1], done, winner
    
    def _process_action(self, agent_id, action, rewards):
        import random
        if agent_id == 1:
            pos = self.agent1_pos
            opp_pos = self.agent2_pos
            cooldown = self.agent1_cooldown
        else:
            pos = self.agent2_pos
            opp_pos = self.agent1_pos
            cooldown = self.agent2_cooldown
        
        if action == 0:  # Up
            pos[0] = max(0, pos[0] - 1)
        elif action == 1:  # Down
            pos[0] = min(self.size - 1, pos[0] + 1)
        elif action == 2:  # Left
            pos[1] = max(0, pos[1] - 1)
        elif action == 3:  # Right
            pos[1] = min(self.size - 1, pos[1] + 1)
        elif action == 4:  # Attack
            if cooldown == 0:
                dist = abs(pos[0] - opp_pos[0]) + abs(pos[1] - opp_pos[1])
                if dist <= 2:
                    damage = 15
                    if agent_id == 1:
                        self.agent2_hp -= damage
                        rewards[0] += 10
                        self.agent1_cooldown = 3
                    else:
                        self.agent1_hp -= damage
                        rewards[1] += 10
                        self.agent2_cooldown = 3
                else:
                    rewards[0 if agent_id == 1 else 1] -= 1
        elif action == 5:  # Use Item
            item = self.agent1_item if agent_id == 1 else self.agent2_item
            if item == 'health':
                if agent_id == 1:
                    self.agent1_hp = min(100, self.agent1_hp + 30)
                    rewards[0] += 5
                    self.agent1_item = None
                else:
                    self.agent2_hp = min(100, self.agent2_hp + 30)
                    rewards[1] += 5
                    self.agent2_item = None
        elif action == 6:  # Dodge
            if cooldown == 0:
                directions = [(-1,0), (1,0), (0,-1), (0,1)]
                dx, dy = random.choice(directions)
                pos[0] = max(0, min(self.size-1, pos[0] + dx))
                pos[1] = max(0, min(self.size-1, pos[1] + dy))
                rewards[0 if agent_id == 1 else 1] += 5
                if agent_id == 1:
                    self.agent1_cooldown = 5
                else:
                    self.agent2_cooldown = 5
        elif action == 7:  # Teleport
            if cooldown == 0 and random.random() < 0.6:
                pos[0] = random.randint(0, self.size-1)
                pos[1] = random.randint(0, self.size-1)
                rewards[0 if agent_id == 1 else 1] += 8
                if agent_id == 1:
                    self.agent1_cooldown = 10
                else:
                    self.agent2_cooldown = 10
        elif action == 8:  # Counter
            if cooldown == 0:
                rewards[0 if agent_id == 1 else 1] += 4
                if agent_id == 1:
                    self.agent1_cooldown = 7
                else:
                    self.agent2_cooldown = 7
        elif action == 9:  # Berserk
            if cooldown == 0:
                rewards[0 if agent_id == 1 else 1] += 6
                if agent_id == 1:
                    self.agent1_cooldown = 15
                else:
                    self.agent2_cooldown = 15
    
    def _check_item_pickup(self, agent_id, rewards):
        pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        pos_tuple = tuple(pos)
        
        if pos_tuple in self.items:
            item = self.items[pos_tuple]
            if agent_id == 1 and self.agent1_item is None:
                self.agent1_item = item
                rewards[0] += 3
                del self.items[pos_tuple]
            elif agent_id == 2 and self.agent2_item is None:
                self.agent2_item = item
                rewards[1] += 3
                del self.items[pos_tuple]
    
    def render(self):
        clear_screen()
        
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}⚔️  AI BATTLE ARENA - LIVE MATCH ⚔️{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        grid = [['·' for _ in range(self.size)] for _ in range(self.size)]
        
        # Place items
        for (x, y), item in self.items.items():
            if item == 'health':
                grid[x][y] = Colors.GREEN + 'H' + Colors.RESET
            elif item == 'attack':
                grid[x][y] = Colors.RED + 'A' + Colors.RESET
            elif item == 'speed':
                grid[x][y] = Colors.YELLOW + 'S' + Colors.RESET
            elif item == 'shield':
                grid[x][y] = Colors.CYAN + 'D' + Colors.RESET
        
        # Place agents (with range indicators)
        a1_x, a1_y = self.agent1_pos
        a2_x, a2_y = self.agent2_pos
        
        # Show attack range
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = a1_x + dx, a1_y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if grid[nx][ny] == '·':
                            grid[nx][ny] = Colors.BLUE + '░' + Colors.RESET
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:
                    nx, ny = a2_x + dx, a2_y + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if grid[nx][ny] == '·' or Colors.BLUE in grid[nx][ny]:
                            grid[nx][ny] = Colors.RED + '░' + Colors.RESET
        
        grid[a1_x][a1_y] = Colors.BLUE + Colors.BOLD + '1' + Colors.RESET
        grid[a2_x][a2_y] = Colors.RED + Colors.BOLD + '2' + Colors.RESET
        
        # Print grid
        print("   " + " ".join([str(i) for i in range(self.size)]))
        for i, row in enumerate(grid):
            print(f" {i} " + " ".join(row))
        
        # Print stats
        print(f"\n{Colors.BOLD}Stats:{Colors.RESET}")
        
        hp1_bar = '█' * (self.agent1_hp // 5) + '░' * (20 - self.agent1_hp // 5)
        hp2_bar = '█' * (self.agent2_hp // 5) + '░' * (20 - self.agent2_hp // 5)
        
        print(f"{Colors.BLUE}Agent 1:{Colors.RESET} [{Colors.GREEN}{hp1_bar}{Colors.RESET}] {self.agent1_hp:3d} HP")
        print(f"         Item: {self.agent1_item or 'None':6s} | Cooldown: {self.agent1_cooldown}")
        print()
        print(f"{Colors.RED}Agent 2:{Colors.RESET} [{Colors.GREEN}{hp2_bar}{Colors.RESET}] {self.agent2_hp:3d} HP")
        print(f"         Item: {self.agent2_item or 'None':6s} | Cooldown: {self.agent2_cooldown}")
        
        print(f"\n{Colors.BOLD}Recent Actions:{Colors.RESET}")
        for log in self.action_log:
            print(f"  {log}")
        
        print(f"\nStep: {self.steps}/{self.max_steps}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")


class BattleCNN(nn.Module):
    def __init__(self, action_size=10):
        super(BattleCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load models
print(f"\n{Colors.BOLD}Loading trained models...{Colors.RESET}\n")

agent1_model = BattleCNN(action_size=10)
agent2_model = BattleCNN(action_size=10)

try:
    checkpoint1 = torch.load('agent1_battle.pth', weights_only=False)
    agent1_model.load_state_dict(checkpoint1['model_state_dict'])
    print(f"{Colors.GREEN}✓{Colors.RESET} Agent 1 loaded")
except:
    print(f"{Colors.RED}✗{Colors.RESET} Agent 1 not found - using random")

try:
    checkpoint2 = torch.load('agent2_battle.pth', weights_only=False)
    agent2_model.load_state_dict(checkpoint2['model_state_dict'])
    print(f"{Colors.GREEN}✓{Colors.RESET} Agent 2 loaded")
except:
    print(f"{Colors.RED}✗{Colors.RESET} Agent 2 not found - using random")

agent1_model.eval()
agent2_model.eval()

print(f"\n{Colors.BOLD}Starting battles in 3 seconds...{Colors.RESET}")
time.sleep(3)

# Run battles
num_battles = 5
agent1_wins = 0
agent2_wins = 0

for battle_num in range(num_battles):
    arena = BattleArena(size=10)
    state1, state2 = arena.reset()
    done = False
    winner = None
    
    while not done:
        arena.render()
        time.sleep(0.15)  # Slow down for visibility
        
        # Get actions
        state1_tensor = torch.FloatTensor(state1)
        state2_tensor = torch.FloatTensor(state2)
        
        with torch.no_grad():
            action1 = agent1_model(state1_tensor).argmax().item()
            action2 = agent2_model(state2_tensor).argmax().item()
        
        state1, state2, _, _, done, winner = arena.step(action1, action2)
    
    # Final render
    arena.render()
    
    if winner == 1:
        agent1_wins += 1
        print(f"\n{Colors.BLUE}{Colors.BOLD}🏆 AGENT 1 WINS! 🏆{Colors.RESET}")
    elif winner == 2:
        agent2_wins += 1
        print(f"\n{Colors.RED}{Colors.BOLD}🏆 AGENT 2 WINS! 🏆{Colors.RESET}")
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}🤝 DRAW! 🤝{Colors.RESET}")
    
    print(f"\nBattle {battle_num + 1}/{num_battles} complete")
    print(f"Score: {Colors.BLUE}Agent 1: {agent1_wins}{Colors.RESET} | {Colors.RED}Agent 2: {agent2_wins}{Colors.RESET}")
    
    if battle_num < num_battles - 1:
        print(f"\nNext battle in 3 seconds...")
        time.sleep(3)

print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
print(f"{Colors.BOLD}FINAL RESULTS{Colors.RESET}")
print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
print(f"{Colors.BLUE}Agent 1 Wins: {agent1_wins}/{num_battles}{Colors.RESET}")
print(f"{Colors.RED}Agent 2 Wins: {agent2_wins}/{num_battles}{Colors.RESET}")

if agent1_wins > agent2_wins:
    print(f"\n{Colors.BLUE}{Colors.BOLD}🎉 AGENT 1 IS THE CHAMPION! 🎉{Colors.RESET}")
elif agent2_wins > agent1_wins:
    print(f"\n{Colors.RED}{Colors.BOLD}🎉 AGENT 2 IS THE CHAMPION! 🎉{Colors.RESET}")
else:
    print(f"\n{Colors.YELLOW}{Colors.BOLD}⚖️  PERFECTLY BALANCED! ⚖️{Colors.RESET}")

print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}\n")
