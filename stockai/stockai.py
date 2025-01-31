import gymnasium as gym
import numpy as np
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import random
import os
import re
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

data = yf.download('AAPL', start='2020-01-01', end='2021-01-01')
stock_data = data['Close'].values
episodes = 101

plt.ion()

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.total_profit = 0
        self.done = False
        self.prev_total_profit = 1000

        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.balance = 1000
        self.shares_held = 0
        self.total_profit = 0
        self.current_step = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        return np.array([self.stock_data[self.current_step]]) / max(self.stock_data)

    def step(self, action):
        current_price = self.stock_data[self.current_step]
        self.current_step += 1

        transaction_cost = 0.01 * current_price  # 1% transaction cost

        # Buy action
        if action == 0:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                self.balance -= transaction_cost  # Include transaction cost

        # Sell action
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                self.balance -= transaction_cost  # Include transaction cost

        # Calculate total asset value and total profit
        total_asset_value = self.balance + (self.shares_held * current_price)
        self.total_profit = total_asset_value - 1000  # Total profit relative to starting balance

        reward = total_asset_value - self.prev_total_profit  # Incremental reward based on profit change
        self.prev_total_profit = total_asset_value  # Update previous total profit

        # Apply a penalty for holding stocks too long
        holding_penalty = -0.01 * self.shares_held  # Penalty for each share held
        reward += holding_penalty

        # Heavier penalty for large negative balances
        if self.total_profit < 0:
            reward -= abs(self.total_profit) * 0.1  # Heavier penalty for large losses

        # End episode if all data is used
        self.done = self.current_step >= len(self.stock_data) - 1
        return self._next_observation(), reward, self.done, {}

env = StockTradingEnv(stock_data)

def create_dqn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape,)),  # Define the input shape explicitly
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(3, activation="linear")  # 3 actions: Buy, Hold, Sell
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_dqn_model(self.state_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) > batch_size:
            self.train(batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

fig, ax = plt.subplots()
total_profits = []
total_rewards = []
line_profit, = ax.plot([], [], label='Total Profit')
line_reward, = ax.plot([], [], label='Total Reward')
plt.xlabel('Episode')
plt.ylabel('Performance')
plt.title('Agent Performance Over Time')
plt.legend()

def update_plot(episode, profits, rewards, total_episodes_run):
    line_profit.set_xdata(np.arange(total_episodes_run - len(profits) + 1, total_episodes_run + 1))
    line_profit.set_ydata(profits)
    
    line_reward.set_xdata(np.arange(total_episodes_run - len(rewards) + 1, total_episodes_run + 1))
    line_reward.set_ydata(rewards)
    
    ax.relim()
    ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.01)

batch_size = 32

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

model_filename = 'stock_trading_agent.keras'
script_directory = os.path.dirname(os.path.abspath(__file__))
models_directory = os.path.join(script_directory, 'Models')

def get_latest_model():
    model_files = [f for f in os.listdir(models_directory) if re.match(r'model_\d+\.keras', f)]
    if not model_files:
        return None  # No models found
    latest_model = max(model_files, key=lambda f: int(re.findall(r'\d+', f)[0]))  # Find highest numbered model
    return latest_model
latest_model_file = get_latest_model()

if latest_model_file:
    model_path = os.path.join(models_directory, latest_model_file)
    print(f"Loading saved model: {model_path}...")
    agent.model = load_model(model_path, compile=False)
    agent.model.compile(loss=MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
else:
    print("No saved model found. Initializing a new model...")
    agent.model = create_dqn_model(input_shape=state_size)
    latest_model_file = "model_0.keras"  # Start from model_0 if no models exist

# Extract the model number for saving the next model
current_model_number = int(re.findall(r'\d+', latest_model_file)[0])
next_model_number = current_model_number + 1
next_model_filename = f"model_{next_model_number}.keras"
next_model_path = os.path.join(models_directory, next_model_filename)
chart_filename = f'training_chart_model_{next_model_number}.png'
chart_path = os.path.join(models_directory, chart_filename)

episode_file = os.path.join(models_directory, 'episode_count.txt')

# Check if the file exists and load the episode count, else initialize to 0
if os.path.exists(episode_file):
    with open(episode_file, 'r') as f:
        total_episodes_run = int(f.read())  # Load the episode count
else:
    # File does not exist, so create it and initialize with 0
    total_episodes_run = 0  # Initialize to 0
    with open(episode_file, 'w') as f:
        f.write(str(total_episodes_run))

print(f"Total episodes previously run: {total_episodes_run}")

total_rewards = []
total_profits = []

for total_episodes_run in range(total_episodes_run, total_episodes_run + episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    episode_reward = 0

    for time in range(len(stock_data)):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            total_profits.append(env.total_profit)
            total_rewards.append(episode_reward)
            if total_episodes_run % 5 == 0:
                update_plot(total_episodes_run, total_profits, total_rewards, total_episodes_run)
            print(f"Episode {total_episodes_run+1}/{total_episodes_run + episodes}, Total Profit: {env.total_profit}, Total Reward: {episode_reward}")
            break

    agent.replay(batch_size)

    if total_episodes_run % 10 == 0:
        with open(episode_file, 'w') as f:
            f.write(str(total_episodes_run))
        if total_episodes_run % 100 == 0:
            agent.model.save(next_model_path)
            print(f"Model saved after episode {total_episodes_run} at {os.path.abspath(next_model_path)}")
    
    total_episodes_run += 1

plt.ioff()
plt.savefig(chart_path)  # Save the chart as a PNG
plt.show()
print(f"Chart saved at {chart_path}")

print("Training complete!")
