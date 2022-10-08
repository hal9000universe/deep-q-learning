# py
from typing import Mapping, Tuple, Callable, Dict

# nn & rl
import gym
import haiku as hk
import optax
from bayes_opt import BayesianOptimization, UtilityFunction

# lib
from Base.q_agent import Agent


class ParamAgent(Agent):

    def __init__(self,
                 network: hk.Transformed,
                 params: hk.Params,
                 optimizer: optax.adam,
                 opt_state: Mapping,
                 env: gym.Env,
                 buffer_size: int,
                 obs_shape: Tuple,
                 ac_shape: Tuple,
                 max_episodes: int,
                 max_steps: int,
                 training_start: int,
                 back_up_frequency: int,
                 reward_to_reach: float,
                 num_actions: int,
                 saving_directory: str,
                 time_episodes: bool = False,
                 time_functions: bool = False,
                 monitoring: bool = False,
                 gamma: float = 0.,
                 epsilon: float = 0.,
                 epsilon_decay_rate: float = 0.,
                 min_epsilon: float = 0.,
                 replace_frequency: int = 0,
                 batch_size: int = 0,
                 train_frequency: int = 0,
                 ):
        super(ParamAgent, self).__init__(
            network,
            params,
            optimizer,
            opt_state,
            env,
            buffer_size,
            obs_shape,
            ac_shape,
            gamma,
            epsilon,
            epsilon_decay_rate,
            min_epsilon,
            max_episodes,
            max_steps,
            training_start,
            batch_size,
            train_frequency,
            back_up_frequency,
            replace_frequency,
            reward_to_reach,
            num_actions,
            saving_directory,
            time_episodes,
            time_functions,
            monitoring,
            verbose=0,
        )

    @property
    def max_episodes(self) -> int:
        return self.max_episodes

    @max_episodes.setter
    def max_episodes(self, value: int):
        self._max_episodes = value

    def inject(self,
               gamma: float,
               epsilon: float,
               epsilon_decay_rate: float,
               min_epsilon: float,
               replace_frequency: int,
               batch_size: int,
               train_frequency: int,
               ):
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._min_epsilon = min_epsilon
        self._replace_frequency = replace_frequency
        self._batch_size = batch_size
        self._train_frequency = train_frequency


def generate_util_func(agent: ParamAgent, episodes) -> Callable:
    agent.max_episodes = episodes

    def util_func(gamma: float,
                  epsilon: float,
                  epsilon_decay_rate: float,
                  min_epsilon: float,
                  replace_frequency: int,
                  batch_size: int,
                  train_frequency: int,
                  ) -> float:
        agent.inject(gamma, epsilon, epsilon_decay_rate, min_epsilon, replace_frequency, batch_size, train_frequency)
        agent.training()
        avg_reward = agent.evaluate()
        return avg_reward

    return util_func


def optimize(agent: ParamAgent, episodes: int = 500, runs: int = 20) -> Dict:
    black_box_func: Callable = generate_util_func(agent, episodes)
    bounds: Dict = {
        "gamma": [0.9, 0.999],
        "epsilon": [0.6, 1.],
        "epsilon_decay_rate": [0.9, 0.999],
        "min_epsilon": [0.001, 0.2],
        "replace_frequency": [20, 70],
        "batch_size": [38, 70],
        "train_frequency": [2, 15]
    }
    optimizer = BayesianOptimization(black_box_func, pbounds=bounds, verbose=2, random_state=1000)
    util_func = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)
    for run in range(runs):
        next_point = optimizer.suggest(util_func)
        next_point["replace_frequency"] = int(next_point["replace_frequency"])
        next_point["batch_size"] = int(next_point["batch_size"])
        next_point["train_frequency"] = int(next_point["train_frequency"])
        target = black_box_func(**next_point)
        optimizer.register(next_point, target)
        print("Run: {}".format(run))
        print("params: {} \n target: {}".format(next_point, target))
        print("-----")
    return optimizer.max
