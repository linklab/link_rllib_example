########## DISCRETE ACTION - ENVIRONMENTS ##########
from ray.tune.registry import register_env
from tutorials.codes.custom_env.tic_tac_toe_343 import TicTacToe343

register_env("TicTacToe343", lambda env_config: TicTacToe343(env_config=env_config))
ENV_NAME = "TicTacToe343"
MAX_TRAIN_ITERATIONS = 200
EPISODE_REWARD_AVG_SOLVED = 1.0
CUSTOM_RAY_CONFIG = {}
ENV_CONFIG = {
    "board_rows": 3,
    "board_cols": 4,
    "mode": 0
}
#######################################
# mode
# 0: 선공 - AI, 후공 - Dummy
# 1: 선공 - Dummy, 후공 - AI
# 2: 선공 - AI, 후공 - AI (Self Play)
#######################################