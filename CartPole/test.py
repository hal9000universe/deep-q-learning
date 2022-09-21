# py
import time
from typing import Callable, Tuple

# nn & rl
import gym
import jax
import haiku as hk
import optax
from numpy import ndarray

# lib
from Base.q_agent import Agent
from Base.utils import generate_loading, generate_visualization
from CartPole.env import ObsWrapper
from CartPole.dqn import Model


if __name__ == '__main__':
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 100000
    MAX_STEPS: int = 3000
    MAX_EPISODES: int = 1000
    REPLACE_FREQUENCY: int = 10
    BACKUP_FREQUENCY: int = 100
    TRAINING_START: int = 100
    TRAIN_FREQUENCY: int = 1
    EPSILON: float = 1.
    EPSILON_DECAY_RATE: float = 1.
    MIN_EPSILON: float = 0.1
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.01
    REWARD_TO_REACH: float = 200.
    DIR: str = "cart_pole"

    env: gym.Env = ObsWrapper(gym.make('CartPole-v1'))
    env.seed(100)
    obs_shape: Tuple = (BUFFER_SIZE, 6)
    ac_shape: Tuple = (BUFFER_SIZE,)
    NUM_ACTIONS: int = env.action_space.n

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: ndarray = env.reset()

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model(NUM_ACTIONS)(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state = optimizer.init(parameters)

    agent = Agent(
        network=model,
        params=parameters,
        optimizer=optimizer,
        opt_state=optimizer_state,
        env=env,
        buffer_size=BUFFER_SIZE,
        obs_shape=obs_shape,
        ac_shape=ac_shape,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay_rate=EPSILON_DECAY_RATE,
        min_epsilon=MIN_EPSILON,
        max_episodes=MAX_EPISODES,
        max_steps=MAX_STEPS,
        training_start=TRAINING_START,
        batch_size=BATCH_SIZE,
        train_frequency=TRAIN_FREQUENCY,
        back_up_frequency=BACKUP_FREQUENCY,
        replace_frequency=REPLACE_FREQUENCY,
        reward_to_reach=REWARD_TO_REACH,
        num_actions=NUM_ACTIONS,
        saving_directory=DIR,
        time_episodes=False,
        time_functions=False,
        monitoring=False,
    )
    agent.training()

    load_state: Callable = generate_loading(DIR)
    visualize: Callable = generate_visualization(env, model)

    parameters, optimizer_state = load_state()
    for _ in range(10):
        visualize(parameters)
