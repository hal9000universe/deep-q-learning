# py
import time
from typing import Tuple, Mapping
from warnings import simplefilter

# nn & rl
import gym
import jax
import haiku as hk
import optax
from numpy import ndarray

# lib
from Base.hyperparameter_optimization import ParamAgent, optimize
from LunarLander.env import ObsWrapper
from LunarLander.dddqn import Model


if __name__ == '__main__':
    simplefilter(action='ignore', category=FutureWarning)

    BATCH_SIZE: int = 32
    BUFFER_SIZE: int = 40000
    MAX_STEPS: int = 1500
    MAX_EPISODES: int = 10000
    REPLACE_FREQUENCY: int = 40
    BACKUP_FREQUENCY: int = 50
    TRAINING_START: int = 500
    TRAIN_FREQUENCY: int = 4
    EPSILON: float = 1.0
    EPSILON_DECAY_RATE: float = 0.995
    MIN_EPSILON: float = 0.1
    GAMMA: float = 0.995
    LEARNING_RATE: float = 0.0001
    TPT_REWARD: float = 220.0
    REWARD_TO_REACH: float = 240.0
    DIR: str = "lunar_lander"

    env: gym.Env = ObsWrapper(gym.make('LunarLander-v2'), MAX_STEPS)
    obs_shape: Tuple = (BUFFER_SIZE, 9)
    ac_shape: Tuple = (BUFFER_SIZE,)
    NUM_ACTIONS: int = env.action_space.n

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: ndarray = env.reset()

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model(NUM_ACTIONS)(*args)))
    optimizer: optax.adam = optax.adam(LEARNING_RATE)

    parameters: hk.Params = model.init(rng, test_input)
    optimizer_state: Mapping = optimizer.init(parameters)

    agent = ParamAgent(
        network=model,
        params=parameters,
        optimizer=optimizer,
        opt_state=optimizer_state,
        env=env,
        buffer_size=BUFFER_SIZE,
        obs_shape=obs_shape,
        ac_shape=ac_shape,
        max_episodes=MAX_EPISODES,
        max_steps=MAX_STEPS,
        training_start=TRAINING_START,
        back_up_frequency=BACKUP_FREQUENCY,
        tpt_reward=TPT_REWARD,
        reward_to_reach=REWARD_TO_REACH,
        num_actions=NUM_ACTIONS,
        saving_directory=DIR,
        time_episodes=False,
        time_functions=False,
        monitoring=False,
    )
    params = optimize(agent)
    print(params)


# hyper-params found: {
# 'target': -0.5397008039787093,
# 'params': {'batch_size': 38.0,
#           'epsilon': 0.7083715136321693,
#           'epsilon_decay_rate': 0.9602451480876931,
#           'gamma': 0.9340376252278284,
#           'min_epsilon': 0.06176775796146821,
#           'replace_frequency': 64.0,
#           'train_frequency': 7.0}
# }
