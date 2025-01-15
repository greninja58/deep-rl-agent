import gymnasium as gym
import numpy as np
import math
from scipy.optimize import minimize
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import turtle
# import wandb
from collections import deque

# 1. Define the Custom Environment
class EquationSolverEnv(gym.Env):
    
    """
    Custom Gymnasium environment to solve the equation p^a + q^b = k using Reinforcement Learning.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, p, q, k, buffer=0.2):
        self.p = p
        self.q = q
        self.k = k
        self.buffer = buffer
        super(EquationSolverEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(8)
        
        # Observation space is a tuple of (a, b, d)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, float('inf')], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.max_steps = 100
        self.current_step = 0

        # Initialize the turtle for rendering
        self.screen = turtle.Screen()
        self.screen.setup(width=700, height=700)
        self.screen.setworldcoordinates(-1, -1, 4, 4)
        self.screen.clear()
        self.draw_graph()
        self.draw_equation_line(self.p,self.q,self.k)
        self.draw_equation_line(self.p+self.buffer,self.q+self.buffer,self.k,"red")
        self.draw_equation_line(self.p-self.buffer,self.q-self.buffer,self.k,"red")
        self.t = turtle.Turtle()
        self.t.penup()
        self.t.speed(0)
        
    def draw_graph(self):
        # Draw grid for visualization
        grid_turtle = turtle.Turtle()
        grid_turtle.speed(0)
        grid_turtle.penup()
        grid_turtle.hideturtle()

        for i in range(-10, 11):
            grid_turtle.goto(-10, i)
            grid_turtle.pendown()
            grid_turtle.goto(10, i)
            grid_turtle.penup()

            grid_turtle.goto(i, -10)
            grid_turtle.pendown()
            grid_turtle.goto(i, 10)
            grid_turtle.penup()

        grid_turtle.goto(-10, 0)
        grid_turtle.pendown()
        grid_turtle.goto(10, 0)
        grid_turtle.penup()

        grid_turtle.goto(0, -10)
        grid_turtle.pendown()
        grid_turtle.goto(0, 10)
        grid_turtle.penup()

    def draw_equation_line(self,p,q,k,color="green"):
        # Draw the line representing p^x + q^y = k
        line_turtle = turtle.Turtle()
        line_turtle.speed(0)
        line_turtle.penup()
        line_turtle.hideturtle()
        line_turtle.color(color)
        line_turtle.pensize(2)

        for x in np.arange(0, k/p+10, 0.05):
            right_side = k - p**x
            if right_side > 0:
                y = round(np.log(right_side) / np.log(q),3)
                line_turtle.goto(x, y)
                line_turtle.pendown()
        line_turtle.pensize(1)

    def get_valid_ab_pair(self):
        min_a = 0.001
        max_a = np.log(self.k) / np.log(self.p)

        while True:
            a = round(random.uniform(min_a, max_a), 3)
            right_side = self.k - self.p**a
            if right_side > 0:
                b = np.log(right_side) / np.log(self.q)
                if b > 0:
                    return a, round(b, 3)

    def reset(self, seed=None, options=None):
        self.t.penup()
        super().reset(seed=seed)
        
        a, b = self.get_valid_ab_pair()

        if random.random() < 1:
            a = round(random.uniform(0, np.log(self.k)/np.log(self.p)), 3)
            b =  round(np.log(self.k - self.p**a) / np.log(self.q), 3)
        d = self.calculate_d(a, b)
        self.state = np.array([a, b, d], dtype=np.float32)
        self.current_step = 0

        self.t.color("grey")
        self.t.goto(round(a,3), round(b,3))
        self.t.pendown()

        return self.state, {}

    def step(self, action):
        self.state = self.fetch_next_state(action)
        a, b, d = self.state
        current_value = self.p**a + self.q**b

        if d < self.buffer:
            reward = 3
            done = False
        else:
            reward = -1 if d > 2 else 1 / 10*(d)
            done = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        self.t.goto(a, b)

        return self.state, reward, done, False, {}

    def fetch_next_state(self, action):
        a, b, _ = self.state

        if action == 0:  
            a += 0.05
        elif action == 1:  
            b += 0.05
        elif action == 2:  
            a = max(a - 0.05, 0)
        elif action == 3:  
            b = max(b - 0.05, 0)
        elif action == 4:
            a += 0.025
            b += 0.025
        elif action == 5:
            a += 0.025
            b -= 0.025
        elif action == 6:
            a -= 0.025
            b += 0.025
        elif action == 7:
            a -= 0.025
            b -= 0.025
        
        d = self.calculate_d(a, b)
        return np.array([a, b, d], dtype=np.float32)

    # def calculate_d(self, a, b):
    #     point = (self.p**a, self.q**b)

    #     def distance_func(xy):
    #         x, y = xy
    #         return np.sqrt((self.p**a - self.p**x)**2 + (self.q**b - self.q**y)**2)

    #     def constraint(xy):
    #         x, y = xy
    #         return self.p**x + self.q**y - self.k

    #     initial_guess = (a, b)
    #     result = minimize(distance_func, initial_guess, constraints={'type': 'eq', 'fun': constraint})

    #     return result.fun
    def calculate_d(self, a, b):
        # Point (a, b) remains unchanged in the distance calculation
        point = (a, b)

        # Distance function calculates Euclidean distance between (a, b) and (x, y)
        def distance_func(xy):
            x, y = xy
            return np.sqrt((a - x)**2 + (b - y)**2)

        # Constraint ensures the point (x, y) lies on the curve p^x + q^y = k
        def constraint(xy):
            x, y = xy
            return self.p**x + self.q**y - self.k

        # Initial guess is still (a, b)
        initial_guess = np.array((a, b), dtype=np.float64)

        
        # Perform the minimization with the constraint
        result = minimize(distance_func, initial_guess, constraints={'type': 'eq', 'fun': constraint})

        # Return the minimized distance
        return result.fun

    def render(self, mode='human'):
        a, b, d = self.state
        print(f"a: {a}, b: {b}, d: {d:.2f}")

    def close(self):
        self.draw_equation_line(self.p+self.buffer,self.q+self.buffer,self.k,"red")
        self.draw_equation_line(self.p-self.buffer,self.q-self.buffer,self.k,"red")
        self.screen.bye()

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.value_fc = nn.Linear(64, 1)
        self.advantage_fc = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        q_values = value + (advantage - advantage.mean())
        return q_values

# Save the trained model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)


# 3. Training Function
def train_dqn(env, num_episodes=500, gamma=0.7, lr=1e-3, batch_size=500, epsilon_start=0.95, epsilon_decay=0.99, epsilon_min=0.1):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the DQN model, target model, loss function, and optimizer
    model = DuelingDQN(state_size, action_size)
    # model = load_model("eq.pth",state_size,action_size)
    target_model = DuelingDQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Experience replay buffer
    memory = deque(maxlen=2000)

    # Training metrics
    rewards = []
    losses = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        episode_loss = 0

        for t in range(env.max_steps):
            # Select action using epsilon-greedy strategy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = model(state).argmax().item()

            # Take action and observe the outcome
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward

            # Store experience in replay buffer
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # Sample a mini-batch from the replay buffer
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.LongTensor(actions)
                rewards_ = torch.FloatTensor(rewards_)
                next_states = torch.stack(next_states)
                dones = torch.FloatTensor(dones)

                # Compute Q-values and target Q-values using Double DQN
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = model(next_states).gather(1, model(next_states).argmax(1).unsqueeze(1)).squeeze(1)
                target_q_values = rewards_ + gamma * target_model(next_states).max(1)[0] * (1 - dones)

                # Compute loss and update the model
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item()

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards.append(total_reward)
        losses.append(episode_loss / (t + 1))

        # Log metrics to WandB
        # wandb.log({
        #     "Episode": episode + 1,
        #     "Total Reward": total_reward,
        #     "Loss": episode_loss / (t + 1),
        #     "Epsilon": epsilon
        # })

        if (episode + 1) % 10 == 0:
            env.render()
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Loss: {episode_loss:.4f}, Epsilon: {epsilon:.2f}")

        # Update target model
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

    save_model(model, "eq.pth")
    # wandb.save('src/main/java/org/cloudsimplus/prototype/custom_eq_model.pth')
    # wandb.watch(model, log="all")
    return rewards, losses


# 4. Plotting Function
def plot_metrics(rewards, losses):
    """
    Plots the training metrics: rewards and losses.
    """
    episodes = range(1, len(rewards) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, label='Total Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()
    plt.grid(True)

    # Plot Losses
    plt.subplot(1, 2, 2)
    plt.plot(episodes, losses, label='Average Loss', color='orange')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Average Loss Over Episodes')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def load_model(filename, state_size, action_size):
    model = DuelingDQN(state_size, action_size)
    model.load_state_dict(torch.load(filename,weights_only=True))
    model.eval()
    return model


def test_agent(env, model, num_episodes=10, epsilon=0.2):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        for t in range(env.max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action = model(state).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward

            state = next_state

            if done:
                break

        total_rewards.append(total_reward)
        # wandb.log({
        #     "Test Episode": episode + 1,
        #     "Test Reward": total_reward
        # })
        env.render()
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")

    return total_rewards


# 5. Main Execution
if __name__ == "__main__":
    # Initialize environment
    env = EquationSolverEnv(11,6,190,.1)
    # # Train the DQN agent
    num_training_episodes = 100 # Adjust as needed
    # wandb.init(project="graph_eqn_dqn", config={
    #     "num_episodes": num_training_episodes,
    #     "gamma": 0.7,
    #     "learning_rate": 1e-3,
    #     "batch_size": 64,
    #     "epsilon_start": .95,
    #     "epsilon_decay": 0.99,
    #     "epsilon_min": 0.01
    # })
    rewards, losses = train_dqn(env, num_episodes=num_training_episodes)

    # # Plot the training metrics
    plot_metrics(rewards, losses)
    env = EquationSolverEnv(11,6,190,.15)
    # Load the trained model
    model = load_model('src/main/java/org/cloudsimplus/prototype/eq.pth', env.observation_space.shape[0], env.action_space.n)

    # Test the trained agent
    test_agent(env, model, num_episodes=10)  # Adjust the number of episodes for testing as needed
    
    # Close the environment
    env.close()
