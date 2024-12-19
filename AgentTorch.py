import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))
    return vec

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    # Add clipping to prevent extreme values
    diffs = np.clip([block[i + 1] - block[i] for i in range(n - 1)], -10, 10)
    res = [sigmoid(diff) for diff in diffs]
    return np.array([res])

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.max_inventory = 5  # Limit maximum inventory size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load("models/" + model_name) if is_eval else self._model()
        
        # Initialize rewards list
        self.rewards = []
        
        # Add training metrics
        self.total_trades = 0
        self.successful_trades = 0

    def _model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            options = self.model(state)
            
        # Add action restrictions based on inventory
        if len(self.inventory) >= self.max_inventory:  # Can't buy if inventory full
            options[0][1] = float('-inf')
        if len(self.inventory) == 0:  # Can't sell if inventory empty
            options[0][2] = float('-inf')
            
        return torch.argmax(options).item()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        mini_batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([transition[0][0] for transition in mini_batch]).to(self.device)
        actions = torch.LongTensor([transition[1] for transition in mini_batch]).to(self.device)
        rewards = torch.FloatTensor([transition[2] for transition in mini_batch]).to(self.device)
        next_states = torch.FloatTensor([transition[3][0] for transition in mini_batch]).to(self.device)
        dones = torch.FloatTensor([transition[4] for transition in mini_batch]).to(self.device)

        # Current Q Values
        current_q_values = self.model(states)
        # Next Q Values
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q = torch.max(next_q_values, 1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

        # Update only the Q values for actions taken
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Compute loss and update
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q_values.unsqueeze(1))
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(stock_name, window_size, episode_count, batch_size):
    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        
        try:
            for t in range(l):
                action = agent.act(state)
                next_state = getState(data, t + 1, window_size + 1)
                reward = 0

                if action == 1 and len(agent.inventory) < agent.max_inventory:  # buy
                    agent.inventory.append(data[t])
                    print("Buy: " + formatPrice(data[t]))
                    agent.total_trades += 1

                elif action == 2 and len(agent.inventory) > 0:  # sell
                    bought_price = agent.inventory.pop(0)
                    reward = max(data[t] - bought_price, 0)
                    total_profit += data[t] - bought_price
                    
                    if data[t] - bought_price > 0:
                        agent.successful_trades += 1
                    
                    print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

                done = t == l - 1
                agent.memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size)

                if t % 100 == 0:  # Print progress every 100 steps
                    print(f"Step {t}/{l}")

            # Episode summary
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print(f"Success Rate: {(agent.successful_trades/max(1, agent.total_trades))*100:.2f}%")
            print("--------------------------------")

            # Save model periodically
            if e % 10 == 0:
                torch.save(agent.model, f"models/model_ep{e}.pth")
                
        except Exception as e:
            print(f"Error in episode {e}: {str(e)}")
            continue

    return agent

# Usage
if __name__ == "__main__":
    stock_name = 'GOLD'
    window_size = 3
    episode_count = 10
    batch_size = 32
    
    agent = train_agent(stock_name, window_size, episode_count, batch_size)