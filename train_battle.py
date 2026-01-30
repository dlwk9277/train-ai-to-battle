import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
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
        # Agent positions
        self.agent1_pos = [0, 0]  # Top-left
        self.agent2_pos = [self.size-1, self.size-1]  # Bottom-right
        
        # Health
        self.agent1_hp = 100
        self.agent2_hp = 100
        
        # Inventory (active item)
        self.agent1_item = None
        self.agent2_item = None
        
        # Item cooldowns
        self.agent1_cooldown = 0
        self.agent2_cooldown = 0
        
        # Items on ground: {(x, y): item_type}
        self.items = {}
        self.spawn_items(3)  # Start with 3 items
        
        self.steps = 0
        self.max_steps = 300
        
        return self.get_state(1), self.get_state(2)
    
    def spawn_items(self, count):
        """Spawn random items on empty tiles"""
        item_types = ['health', 'attack', 'speed', 'shield']
        
        for _ in range(count):
            while True:
                x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
                if (x, y) not in self.items and \
                   [x, y] != self.agent1_pos and \
                   [x, y] != self.agent2_pos:
                    self.items[(x, y)] = random.choice(item_types)
                    break
    
    def get_state(self, agent_id):
        """Create state representation for an agent"""
        state = np.zeros((self.size, self.size, 6))
        
        # Channel 0: Own position
        if agent_id == 1:
            state[self.agent1_pos[0], self.agent1_pos[1], 0] = 1
        else:
            state[self.agent2_pos[0], self.agent2_pos[1], 0] = 1
        
        # Channel 1: Opponent position
        if agent_id == 1:
            state[self.agent2_pos[0], self.agent2_pos[1], 1] = 1
        else:
            state[self.agent1_pos[0], self.agent1_pos[1], 1] = 1
        
        # Channel 2: Health items
        # Channel 3: Attack items
        # Channel 4: Speed items
        # Channel 5: Shield items
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
        """
        Actions:
        0-3: Move (Up, Down, Left, Right)
        4: Attack
        5: Use Item
        6: SECRET - Dodge (evade next attack)
        7: SECRET - Teleport (random location)
        8: SECRET - Counter (reflect damage)
        9: SECRET - Berserk (2x damage for 3 turns)
        """
        self.steps += 1
        rewards = [0, 0]
        
        # Decrease cooldowns
        self.agent1_cooldown = max(0, self.agent1_cooldown - 1)
        self.agent2_cooldown = max(0, self.agent2_cooldown - 1)
        
        # Process movements and basic actions simultaneously
        self._process_action(1, action1, rewards)
        self._process_action(2, action2, rewards)
        
        # Check for item pickups
        self._check_item_pickup(1, rewards)
        self._check_item_pickup(2, rewards)
        
        # Spawn new items occasionally
        if self.steps % 20 == 0 and len(self.items) < 5:
            self.spawn_items(1)
        
        # Small step penalty
        rewards[0] -= 0.1
        rewards[1] -= 0.1
        
        # Check win conditions
        done = False
        if self.agent1_hp <= 0:
            rewards[0] -= 200
            rewards[1] += 200
            done = True
        elif self.agent2_hp <= 0:
            rewards[0] += 200
            rewards[1] -= 200
            done = True
        elif self.steps >= self.max_steps:
            # Winner is whoever has more HP
            if self.agent1_hp > self.agent2_hp:
                rewards[0] += 100
                rewards[1] -= 100
            elif self.agent2_hp > self.agent1_hp:
                rewards[0] -= 100
                rewards[1] += 100
            done = True
        
        return self.get_state(1), self.get_state(2), rewards[0], rewards[1], done
    
    def _process_action(self, agent_id, action, rewards):
        """Process action for one agent"""
        if agent_id == 1:
            pos = self.agent1_pos
            opp_pos = self.agent2_pos
            cooldown = self.agent1_cooldown
        else:
            pos = self.agent2_pos
            opp_pos = self.agent1_pos
            cooldown = self.agent2_cooldown
        
        # Movement (0-3)
        if action == 0:  # Up
            pos[0] = max(0, pos[0] - 1)
        elif action == 1:  # Down
            pos[0] = min(self.size - 1, pos[0] + 1)
        elif action == 2:  # Left
            pos[1] = max(0, pos[1] - 1)
        elif action == 3:  # Right
            pos[1] = min(self.size - 1, pos[1] + 1)
        
        # Attack (4)
        elif action == 4:
            if cooldown == 0:
                dist = abs(pos[0] - opp_pos[0]) + abs(pos[1] - opp_pos[1])
                if dist <= 2:  # Attack range
                    damage = 15
                    if agent_id == 1:
                        self.agent2_hp -= damage
                        rewards[0] += 10  # Reward for hitting
                        self.agent1_cooldown = 3
                    else:
                        self.agent1_hp -= damage
                        rewards[1] += 10
                        self.agent2_cooldown = 3
                else:
                    rewards[0 if agent_id == 1 else 1] -= 1  # Penalty for missing
        
        # Use Item (5)
        elif action == 5:
            item = self.agent1_item if agent_id == 1 else self.agent2_item
            if item:
                if item == 'health':
                    if agent_id == 1:
                        self.agent1_hp = min(100, self.agent1_hp + 30)
                        rewards[0] += 5
                        self.agent1_item = None
                    else:
                        self.agent2_hp = min(100, self.agent2_hp + 30)
                        rewards[1] += 5
                        self.agent2_item = None
                
                elif item == 'attack':
                    # Next attack does double damage (simplified)
                    if agent_id == 1:
                        rewards[0] += 3
                        self.agent1_item = None
                    else:
                        rewards[1] += 3
                        self.agent2_item = None
        
        # SECRET: Dodge (6)
        elif action == 6:
            if cooldown == 0:
                # Teleport 1 space in random direction
                directions = [(-1,0), (1,0), (0,-1), (0,1)]
                dx, dy = random.choice(directions)
                pos[0] = max(0, min(self.size-1, pos[0] + dx))
                pos[1] = max(0, min(self.size-1, pos[1] + dy))
                rewards[0 if agent_id == 1 else 1] += 5  # Reward for using secret
                if agent_id == 1:
                    self.agent1_cooldown = 5
                else:
                    self.agent2_cooldown = 5
        
        # SECRET: Teleport (7)
        elif action == 7:
            if cooldown == 0 and random.random() < 0.6:  # 60% success
                pos[0] = random.randint(0, self.size-1)
                pos[1] = random.randint(0, self.size-1)
                rewards[0 if agent_id == 1 else 1] += 8  # Big reward
                if agent_id == 1:
                    self.agent1_cooldown = 10
                else:
                    self.agent2_cooldown = 10
        
        # SECRET: Counter (8)
        elif action == 8:
            if cooldown == 0:
                # Reflect some damage (simplified - just give reward)
                rewards[0 if agent_id == 1 else 1] += 4
                if agent_id == 1:
                    self.agent1_cooldown = 7
                else:
                    self.agent2_cooldown = 7
        
        # SECRET: Berserk (9)
        elif action == 9:
            if cooldown == 0:
                # Double damage mode (simplified - just reward)
                rewards[0 if agent_id == 1 else 1] += 6
                if agent_id == 1:
                    self.agent1_cooldown = 15
                else:
                    self.agent2_cooldown = 15
    
    def _check_item_pickup(self, agent_id, rewards):
        """Check if agent picked up an item"""
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
        """Render the arena to terminal"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
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
        
        # Place agents
        grid[self.agent1_pos[0]][self.agent1_pos[1]] = Colors.BLUE + Colors.BOLD + '1' + Colors.RESET
        grid[self.agent2_pos[0]][self.agent2_pos[1]] = Colors.RED + Colors.BOLD + '2' + Colors.RESET
        
        # Print grid
        print("  " + "".join([str(i) for i in range(self.size)]))
        for i, row in enumerate(grid):
            print(f"{i} " + "".join(row))
        
        # Print stats
        print(f"\n{Colors.BLUE}Agent 1:{Colors.RESET} HP={self.agent1_hp:3d} Item={self.agent1_item or 'None':6s} CD={self.agent1_cooldown}")
        print(f"{Colors.RED}Agent 2:{Colors.RESET} HP={self.agent2_hp:3d} Item={self.agent2_item or 'None':6s} CD={self.agent2_cooldown}")
        print(f"\nStep: {self.steps}/{self.max_steps}")


class BattleCNN(nn.Module):
    """CNN for processing arena state"""
    
    def __init__(self, action_size=10):
        super(BattleCNN, self).__init__()
        
        # Input: 10x10x6
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)  # -> 32x10x10
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # -> 64x10x10
        self.pool = nn.MaxPool2d(2, 2)  # -> 64x5x5
        
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


class BattleAgent:
    """DQN agent for battle arena"""
    
    def __init__(self, agent_id, action_size=10):
        self.agent_id = agent_id
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.learning_rate = 0.0005
        
        self.model = BattleCNN(action_size)
        self.target_model = BattleCNN(action_size)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        self.steps = 0
        self.train_steps = 0
        self.target_update_freq = 20
        
        # Track action usage
        self.action_counts = {i: 0 for i in range(action_size)}
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        self.steps += 1
        
        if training and np.random.rand() <= self.epsilon:
            # 20% chance to try secret actions during exploration
            if np.random.rand() < 0.2:
                action = random.randint(6, 9)
            else:
                action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            action = q_values.argmax().item()
        
        self.action_counts[action] += 1
        return action
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([s for s, a, r, ns, d in minibatch]))
        actions = torch.LongTensor([a for s, a, r, ns, d in minibatch])
        rewards = torch.FloatTensor([r for s, a, r, ns, d in minibatch])
        next_states = torch.FloatTensor(np.array([ns for s, a, r, ns, d in minibatch]))
        dones = torch.FloatTensor([d for s, a, r, ns, d in minibatch])
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_model()
        
        return loss.item()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# TRAINING
print("\n" + "="*70)
print(f"{Colors.BOLD}⚔️  AI BATTLE ARENA - TRAINING ⚔️{Colors.RESET}")
print("="*70)
print("\nTWO AIs will fight for 10,000 episodes!")
print("\nACTIONS:")
print("  [0-3] Move (Up, Down, Left, Right)")
print("  [4]   Attack (range 2, damage 15)")
print("  [5]   Use Item")
print(f"\n{Colors.YELLOW}SECRET ACTIONS (AI must discover!):{Colors.RESET}")
print("  [6] 🏃 Dodge    - Quick escape (+5 reward)")
print("  [7] ⚡ Teleport - Random location (+8 reward, 60% success)")
print("  [8] 🛡️  Counter  - Reflect damage (+4 reward)")
print("  [9] 💥 Berserk  - 2x damage mode (+6 reward)")
print("\n" + "="*70 + "\n")

episodes = 10000
batch_size = 64
arena = BattleArena(size=10)

agent1 = BattleAgent(agent_id=1, action_size=10)
agent2 = BattleAgent(agent_id=2, action_size=10)

agent1_wins = 0
agent2_wins = 0
draws = 0

for episode in range(episodes):
    state1, state2 = arena.reset()
    done = False
    
    episode_reward1 = 0
    episode_reward2 = 0
    
    while not done:
        action1 = agent1.act(state1, training=True)
        action2 = agent2.act(state2, training=True)
        
        next_state1, next_state2, reward1, reward2, done = arena.step(action1, action2)
        
        agent1.remember(state1, action1, reward1, next_state1, done)
        agent2.remember(state2, action2, reward2, next_state2, done)
        
        state1 = next_state1
        state2 = next_state2
        
        episode_reward1 += reward1
        episode_reward2 += reward2
        
        # Train both agents
        if len(agent1.memory) >= batch_size:
            agent1.replay(batch_size)
        if len(agent2.memory) >= batch_size:
            agent2.replay(batch_size)
    
    # Decay epsilon
    agent1.decay_epsilon()
    agent2.decay_epsilon()
    
    # Track winner
    if arena.agent1_hp > arena.agent2_hp:
        agent1_wins += 1
    elif arena.agent2_hp > arena.agent1_hp:
        agent2_wins += 1
    else:
        draws += 1
    
    # Print progress
    if (episode + 1) % 500 == 0:
        print(f"\n{'='*70}")
        print(f"{Colors.BOLD}Episode {episode + 1}/{episodes}{Colors.RESET}")
        print(f"{'='*70}")
        print(f"Agent 1: ε={agent1.epsilon:.3f} | Wins={agent1_wins:4d} ({agent1_wins/(episode+1)*100:5.1f}%)")
        print(f"Agent 2: ε={agent2.epsilon:.3f} | Wins={agent2_wins:4d} ({agent2_wins/(episode+1)*100:5.1f}%)")
        print(f"Draws: {draws}")
        
        # Action distribution
        print(f"\n{Colors.CYAN}Action Usage:{Colors.RESET}")
        action_names = ['Up', 'Down', 'Left', 'Right', 'Attack', 'Use', 
                       '🏃Dodge', '⚡Tele', '🛡️Counter', '💥Berserk']
        
        total1 = sum(agent1.action_counts.values())
        total2 = sum(agent2.action_counts.values())
        
        print(f"\n{Colors.BLUE}Agent 1:{Colors.RESET}")
        for i in range(6, 10):  # Just show secrets
            count = agent1.action_counts[i]
            pct = (count / total1 * 100) if total1 > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  [{i}] {action_names[i]:10s}: {bar:30s} {pct:5.2f}% ({count})")
        
        print(f"\n{Colors.RED}Agent 2:{Colors.RESET}")
        for i in range(6, 10):
            count = agent2.action_counts[i]
            pct = (count / total2 * 100) if total2 > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  [{i}] {action_names[i]:10s}: {bar:30s} {pct:5.2f}% ({count})")
        
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

print("\n" + "="*70)
print(f"{Colors.GREEN}✓ Training Complete!{Colors.RESET}")
print("="*70 + "\n")

# Save models
torch.save({
    'model_state_dict': agent1.model.state_dict(),
    'target_model_state_dict': agent1.target_model.state_dict(),
    'epsilon': agent1.epsilon,
    'action_counts': agent1.action_counts
}, 'agent1_battle.pth')

torch.save({
    'model_state_dict': agent2.model.state_dict(),
    'target_model_state_dict': agent2.target_model.state_dict(),
    'epsilon': agent2.epsilon,
    'action_counts': agent2.action_counts
}, 'agent2_battle.pth')

print(f"✓ Models saved: agent1_battle.pth, agent2_battle.pth")
print(f"✓ Run 'python watch_battle.py' to see them fight!\n")
