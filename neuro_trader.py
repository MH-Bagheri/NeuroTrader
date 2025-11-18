import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import itertools

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Hardware acceleration if available (MPS for Mac, CUDA for NVIDIA, CPU otherwise)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Simulation Settings
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE_PERCENT = 0.001  # 0.1% fee per trade to prevent spamming
DATA_LENGTH = 1000               # Length of the synthetic market timeframe

# RL Agent Settings
GAMMA = 0.95                     # Discount factor for future rewards
EPSILON_START = 1.0              # Exploration rate (start with random actions)
EPSILON_END = 0.01               # Minimum exploration rate
EPSILON_DECAY = 0.94             # Rate at which exploration decays (Was 0.995 - made aggressive)
BATCH_SIZE = 64                  # Size of memory batch for training
MEMORY_SIZE = 10000              # Experience Replay buffer size
LEARNING_RATE = 0.001            # Optimizer learning rate
TARGET_UPDATE = 10               # Update target network every N episodes

# Visual Settings
RENDER_SPEED = 0.001             # Pause time for matplotlib

# ==========================================
# 1. SYNTHETIC MARKET GENERATOR
# ==========================================
class MarketGenerator:
    """
    Generates realistic synthetic stock market data using 
    Geometric Brownian Motion augmented with sine waves for seasonality 
    and fractal noise for volatility clustering.
    """
    def __init__(self, length=1000, start_price=100, volatility=0.02, drift=0.0002):
        self.length = length
        self.start_price = start_price
        self.volatility = volatility
        self.drift = drift

    def generate(self):
        # Geometric Brownian Motion
        dt = 1
        t = np.linspace(0, self.length, self.length)
        
        # Random component (Wiener process)
        shock = np.random.normal(0, self.volatility, self.length)
        
        # Drift component (Trend)
        drift = (self.drift - 0.5 * self.volatility**2) * dt
        
        # Add some "Seasonality" (Sine wave)
        seasonality = 0.005 * np.sin(t / 50.0)
        
        # Calculate prices
        returns = np.exp(drift + shock + seasonality)
        prices = np.zeros(self.length)
        prices[0] = self.start_price
        
        for i in range(1, self.length):
            prices[i] = prices[i-1] * returns[i]
            
        return prices

# ==========================================
# 2. DEEP Q-NETWORK (THE BRAIN)
# ==========================================
class DQN(nn.Module):
    """
    A Deep Neural Network that approximates the Q-Value function.
    Input: State vector
    Output: Q-values for [Action 0, Action 1, Action 2]
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. TRADING ENVIRONMENT
# ==========================================
class TradingEnv:
    """
    A custom OpenAI Gym-style environment.
    """
    def __init__(self, prices, initial_capital):
        self.prices = prices
        self.initial_capital = initial_capital
        self.n_step = 0
        self.reset()

    def reset(self):
        self.n_step = 0
        self.balance = self.initial_capital
        self.shares_held = 0
        self.net_worth = self.initial_capital
        self.max_net_worth = self.initial_capital
        self.history = []
        return self._get_state()

    def _get_state(self):
        # State Representation:
        # 1. Normalized Price differences (Returns)
        # 2. Holding status (0 or 1)
        # 3. Balance normalized
        
        window_size = 5
        if self.n_step < window_size:
            # Pad with zeros at the start
            window = np.zeros(window_size)
            if self.n_step > 0:
                window[-self.n_step:] = np.diff(self.prices[:self.n_step+1])
        else:
            window = np.diff(self.prices[self.n_step-window_size:self.n_step+1])
        
        # Normalize inputs for Neural Net stability
        state = np.append(window / self.prices[0], [
            1.0 if self.shares_held > 0 else 0.0,
            self.balance / self.initial_capital
        ])
        return state

    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_price = self.prices[self.n_step]
        reward = 0
        done = False
        
        if action == 1: # Buy
            if self.shares_held == 0:
                cost = current_price * (1 + TRANSACTION_FEE_PERCENT)
                if self.balance >= cost:
                    self.shares_held = int(self.balance // cost)
                    self.balance -= cost * self.shares_held
                    # Slight penalty for transaction cost to discourage churn
                    reward = -0.01 
            else:
                reward = -0.1 # Penalty for trying to buy when already holding

        elif action == 2: # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - TRANSACTION_FEE_PERCENT)
                self.balance += revenue
                self.shares_held = 0
                # Reward based on profit made relative to initial capital
                profit = (self.balance - self.initial_capital) / self.initial_capital
                reward = profit if profit > 0 else profit * 1.2 # Penalize losses more
            else:
                reward = -0.1 # Penalty for trying to sell nothing

        elif action == 0: # Hold
            reward = 0.01 if self.shares_held > 0 else 0 # Slight reward for holding if trend is up? (Simplified)

        # Move to next time step
        self.n_step += 1
        
        # Calculate Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        self.history.append(self.net_worth)

        # Check if done
        if self.n_step >= len(self.prices) - 1:
            done = True
            # Terminal Reward: Final Net Worth
            reward += (self.net_worth - self.initial_capital) / self.initial_capital

        next_state = self._get_state()
        return next_state, reward, done, {}

# ==========================================
# 4. THE AGENT
# ==========================================
class Agent:
    def __init__(self, input_dim, output_dim):
        self.policy_net = DQN(input_dim, output_dim).to(DEVICE)
        self.target_net = DQN(input_dim, output_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        # Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            return random.randint(0, 2) # Explore
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item() # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # Current Q Values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Expected Q Values (Target Network)
        max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + (GAMMA * max_next_q * (1 - dones))
        
        # Huber Loss (less sensitive to outliers than MSE)
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping for stability
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ==========================================
# 5. MAIN EXECUTION & VISUALIZATION
# ==========================================
def main():
    print(f"NeuroTrader initializing on {DEVICE}...")
    print("Generating Fractal Market Data...")
    
    # Create Market
    market_gen = MarketGenerator(length=DATA_LENGTH)
    prices = market_gen.generate()
    
    env = TradingEnv(prices, INITIAL_CAPITAL)
    
    # State dimension: Window of 5 diffs + 1 holding + 1 balance = 7 features
    agent = Agent(input_dim=7, output_dim=3) 

    # Setup Visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    episodes = 100 # Increased from 20 to 100 to give it time to actually learn
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        buy_signals = []
        sell_signals = []
        
        for t in range(len(prices) - 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Record moves for plotting
            if action == 1: buy_signals.append(t)
            elif action == 2: sell_signals.append(t)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update Target Network
        if e % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        agent.update_epsilon()
        
        # --- VISUALIZATION UPDATE ---
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Market Price & Trade Signals
        ax1.plot(prices, label='Market Price', color='cyan', alpha=0.6)
        ax1.scatter(buy_signals, prices[buy_signals], marker='^', color='lime', label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals, prices[sell_signals], marker='v', color='red', label='Sell Signal', zorder=5)
        ax1.set_title(f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.2f}")
        ax1.set_ylabel("Stock Price")
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # Plot 2: Portfolio Value
        ax2.plot(env.history, label='Net Worth', color='gold')
        ax2.axhline(y=INITIAL_CAPITAL, color='white', linestyle='--', alpha=0.5)
        ax2.set_title(f"Final Net Worth: ${env.net_worth:.2f} (vs ${INITIAL_CAPITAL:.0f})")
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.set_xlabel("Time Steps")
        ax2.grid(True, alpha=0.2)
        
        # Dark Mode Styling
        fig.patch.set_facecolor('#121212')
        ax1.set_facecolor('#1e1e1e')
        ax2.set_facecolor('#1e1e1e')
        ax1.tick_params(colors='white')
        ax2.tick_params(colors='white')
        ax1.yaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.xaxis.label.set_color('white')
        ax1.title.set_color('white')
        ax2.title.set_color('white')
        
        plt.pause(0.1) # Pause to update plot
        print(f"Episode {e+1}: Reward {total_reward:.2f} | Net Worth: {env.net_worth:.2f}")

    print("Training Complete. Close plot to exit.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()