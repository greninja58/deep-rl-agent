import pytest
import torch
import numpy as np
from main import EquationSolverEnv, DuelingDQN, train_dqn, test_agent  # Import your classes and functions

# Test 1: Environment Initialization
def test_environment_initialization():
    env = EquationSolverEnv(p=2, q=3, k=10, buffer=0.1)
    assert env.action_space.n == 8, "Action space should have 8 discrete actions."
    assert len(env.reset()) == 3, "Observation space should return 3 variables: a, b, and d."

# Test 2: Environment Step Function
def test_environment_step():
    env = EquationSolverEnv(p=2, q=3, k=10, buffer=0.1)
    state = env.reset()
    next_state, reward, done, _ = env.step(0)  # Take the first action
    assert isinstance(next_state, np.ndarray), "Next state should be a numpy array."
    assert isinstance(reward, float), "Reward should be a float."
    assert isinstance(done, bool), "Done should be a boolean."

# Test 3: Dueling DQN Initialization
def test_dueling_dqn_initialization():
    input_dim = 3
    output_dim = 8
    model = DuelingDQN(input_dim, output_dim)
    sample_input = torch.rand((1, input_dim))
    output = model(sample_input)
    assert output.shape == (1, output_dim), "Model output shape should match the action space."

# Test 4: Training Function
def test_training_function():
    env = EquationSolverEnv(p=2, q=3, k=10, buffer=0.1)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = DuelingDQN(input_dim, output_dim)
    try:
        train_dqn(
            env=env,
            model=model,
            num_episodes=2,  # Short number of episodes for testing
            gamma=0.99,
            lr=0.001,
            batch_size=32,
            epsilon_start=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.1
        )
    except Exception as e:
        pytest.fail(f"Training function failed with exception: {e}")

# Test 5: Testing Function
def test_testing_function():
    env = EquationSolverEnv(p=2, q=3, k=10, buffer=0.1)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = DuelingDQN(input_dim, output_dim)
    try:
        rewards = test_agent(env=env, model=model, num_episodes=2)
        assert isinstance(rewards, list), "Test function should return a list of rewards."
    except Exception as e:
        pytest.fail(f"Testing function failed with exception: {e}")

# Test 6: Reward Signal
def test_reward_signal():
    env = EquationSolverEnv(p=2, q=3, k=10, buffer=0.1)
    env.reset()
    _, reward, _, _ = env.step(0)
    assert isinstance(reward, float), "Reward should be a float."
    assert reward <= 1.0, "Reward should not exceed 1.0."
