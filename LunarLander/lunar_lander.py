# py
from typing import Any

# nn & rl
import numpy as np
from numpy import append, newaxis, zeros

# custom
from Base.q_agent import *


class ObsWrapper(gym.Wrapper):

    def __init__(self, environment):
        super(ObsWrapper, self).__init__(environment)
        self._step = 0
        self._observation_space = gym.spaces.Box(shape=(1, 9), low=float('-inf'), high=float('inf'))

    def observation(self, observation: ndarray) -> ndarray:
        fraction_finished: float = self._step / MAX_STEPS
        return append(observation, fraction_finished)[newaxis, ...]

    def step(self, action) -> Tuple[ndarray, float, bool, Any]:
        self._step += 1
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs) -> ndarray:
        self._step = 0
        state: ndarray = self.env.reset()
        return self.observation(state)


class Model(hk.Module):
    _lin1: hk.Linear
    _lin2: hk.Linear
    _val: hk.Linear
    _adv: hk.Linear

    def __init__(self):
        super().__init__()
        self._lin1 = hk.Linear(64)
        self._lin2 = hk.Linear(64)
        self._val = hk.Linear(1)
        self._adv = hk.Linear(env.action_space.n)

    def __call__(self, x: ndarray or jnp.ndarray) -> ndarray or jnp.ndarray:
        x = self._lin1(x)
        x = jax.nn.relu(x)
        x = self._lin2(x)
        x = jax.nn.relu(x)
        val: ndarray or jnp.ndarray = self._val(x)
        adv: ndarray or jnp.ndarray = self._adv(x)
        Q: ndarray or jnp.ndarray = val + adv - jnp.mean(adv, axis=1, keepdims=True)
        return Q


def visualize_agent():
    state: ndarray = env.reset()
    for step in range(MAX_STEPS):
        action: int = int(argmax(model.apply(parameters, state)))
        state, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    BATCH_SIZE: int = 64
    BUFFER_SIZE: int = 100000
    MAX_STEPS: int = 1000
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

    env: gym.Env = ObsWrapper(gym.make('LunarLander-v2'))

    rng: jax.random.PRNGKeyArray = jax.random.PRNGKey(time.time_ns())
    test_input: np.ndarray = zeros((1, 9))

    model: hk.Transformed = hk.without_apply_rng(hk.transform(lambda *args: Model()(*args)))
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
        obs_placeholder_shape=(BUFFER_SIZE, 9),
        ac_placeholder_shape=(BUFFER_SIZE,),
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
    )
    agent.training()

    parameters = load_state()
    visualize_agent()
