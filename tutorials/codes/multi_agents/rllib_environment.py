########## DISCRETE ACTION - ENVIRONMENTS ##########
from ray.tune import register_env
from tutorials.codes.custom_env.tic_tac_toe_343 import TicTacToe343


register_env("TicTacToe343", lambda env_config: TicTacToe343(env_config=env_config))
ENV_CONFIG = {}
ENV_NAME = "TicTacToe343"
MAX_TRAIN_ITERATIONS = 100
EPISODE_REWARD_AVG_SOLVED = 1.0
