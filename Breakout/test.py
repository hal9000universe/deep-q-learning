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
from Breakout.env import create_env
from Breakout.dddqn import Model


if __name__ == '__main__':
    BATCH_SIZE: int = 32
    BUFFER_SIZE: int = 100000
    MAX_STEPS: int = 10000
    MAX_EPISODES: int = 10000
    REPLACE_FREQUENCY: int = 50
    BACKUP_FREQUENCY: int = 20
    TRAINING_START: int = 256
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.001
    GAMMA: float = 0.999
    LEARNING_RATE: float = 0.001
    REWARD_TO_REACH: float = 100.
    DIR: str = "breakout"

    env: gym.Env = create_env()
    NUM_ACTIONS: int = env.action_space.n
    OBS_SHAPE: Tuple = (BUFFER_SIZE, 4, 84, 84)
    AC_SHAPE: Tuple = (BUFFER_SIZE,)

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
        obs_shape=OBS_SHAPE,
        ac_shape=AC_SHAPE,
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
    )
    agent.training()

    load_state: Callable = generate_loading(DIR)
    visualize: Callable = generate_visualization(env, model)

    parameters = load_state()
    visualize(parameters)
