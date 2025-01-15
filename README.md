# Deep RL Agent

This **Deep RL Agent**, demonstrates a custom reinforcement learning environment and agent built using PyTorch and Gymnasium to solve mathematical equations of the form:

\[ p^a + q^b = k \]

The agent uses a Dueling Deep Q-Network (Dueling DQN) to explore and learn optimal solutions for this equation through a dynamic and adaptive decision-making process.

## Repository Overview

### Main Components

1. **Custom Environment**
   - The custom environment `EquationSolverEnv` extends `gym.Env` and represents the RL task of solving the equation.
   - The environment features:
     - Action space with 8 discrete actions for incrementing or decrementing variables \(a\) and \(b\) or applying combined modifications.
     - Observation space consisting of three variables: \(a\), \(b\), and \(d\) (distance to the solution curve).
     - Visual rendering using the `turtle` graphics library to plot the agent's progress and the solution curve.

2. **Reinforcement Learning Model**
   - A Dueling Deep Q-Network (Dueling DQN) is implemented using PyTorch, with separate streams for estimating the value function and action advantages.

3. **Training and Testing**
   - Training the DQN agent to learn optimal policies for solving the equation.
   - Testing the trained agent to evaluate its performance over a set of episodes.

4. **Metrics Visualization**
   - Rewards and losses during training are plotted to analyze the agent's learning progress and convergence trends.
   - The metrics provide insights into the effectiveness of the agent's decision-making and the stability of training, enabling performance tuning and evaluation.

### File Structure

- **`main.py`**: The main script containing the custom environment, DQN implementation, training, and testing logic.
- **`src/main/java/org/cloudsimplus/prototype/eq.pth`**: Path to save/load the trained model.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - `gymnasium`
  - `numpy`
  - `torch`
  - `matplotlib`
  - `scipy`
  - `turtle`

Install the dependencies using the following command:

```bash
pip install gymnasium numpy torch matplotlib scipy
```

### Running the Code

1. Clone the repository:


2. Run the main script:

   ```bash
   python main.py
   ```

### Outputs

- **Training Phase**:
  - Logs rewards and losses for each episode.
  - Periodically renders the agent's progress in the turtle graphics window.

- **Testing Phase**:
  - Evaluates the trained agent on new episodes and visualizes its performance.
  - Displays the average reward achieved during testing.

- **Plots**:
  - Total rewards and average losses per episode are plotted after training.

---

## Key Features

### 1. Custom RL Environment
- A tailored Gymnasium environment specifically for solving \(p^a + q^b = k\).
- Visual feedback using `turtle` graphics to observe the agent's exploration and learning.
- Configurable parameters like \(p\), \(q\), \(k\), and buffer size for customization.

### 2. Dueling DQN Implementation
- A Dueling Deep Q-Network with:
  - Fully connected layers for processing state inputs.
  - Separate streams for value function and advantage computation.

### 3. Training & Testing Workflow
- Epsilon-greedy exploration strategy with adjustable decay and minimum thresholds.
- Experience replay buffer for stable training.
- Target network updates to improve learning stability.

---

## Functions and Usage

### Environment
- **Class**: `EquationSolverEnv`
- **Constructor Parameters**:
  - `p`: Base for the first term.
  - `q`: Base for the second term.
  - `k`: Target value.
  - `buffer`: Distance threshold for considering solutions as acceptable.

### Training
- **Function**: `train_dqn`
- **Parameters**:
  - `env`: The RL environment.
  - `num_episodes`: Number of training episodes.
  - `gamma`: Discount factor.
  - `lr`: Learning rate.
  - `batch_size`: Size of the experience replay batch.
  - `epsilon_start`, `epsilon_decay`, `epsilon_min`: Epsilon-greedy exploration parameters.

### Testing
- **Function**: `test_agent`
- **Parameters**:
  - `env`: The RL environment.
  - `model`: Trained DQN model.
  - `num_episodes`: Number of test episodes.

---

## Future Enhancements

1. **Enhanced Visualizations**:
   - Real-time performance metrics during training.
   - 3D plots of the solution space.

2. **Extended Action Space**:
   - Introduce more granular modifications to \(a\) and \(b\).

3. **Integration with Cloud Platforms**:
   - Extend the environment to model real-world cloud-scaling problems based on mathematical equations.

---

## Contributions

Feel free to open issues or create pull requests for bug fixes, feature enhancements, or other suggestions.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

For inquiries or support, contact the repository maintainer via [email address or GitHub profile link].

