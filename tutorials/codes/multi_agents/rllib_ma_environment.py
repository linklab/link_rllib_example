########## DISCRETE ACTION - ENVIRONMENTS ##########
from ray.rllib.env import Unity3DEnv
from ray.tune.registry import register_env
from tutorials.codes.custom_env.tic_tac_toe_343 import TicTacToe343

register_env("TicTacToe343", lambda env_config: TicTacToe343(env_config=env_config))
ENV_NAME = "TicTacToe343"
MAX_TRAIN_ITERATIONS = 200
EPISODE_REWARD_AVG_SOLVED = 1.0
ENV_CONFIG = {
    "board_rows": 3,
    "board_cols": 4
}
MODE = 0
# 0: 선공 - AI, 후공 - Dummy
# 1: 선공 - Dummy, 후공 - AI
# 2: 선공 - AI, 후공 - AI
# 3: 선공 - AI, 후공 - AI (Self Play)


################
# env_name = "/Users/zero/PycharmProjects/link_rllib_example/unity_env/3DBall_Darwin"
# env_config={
#     "file_name": env_name,
#     "episode_horizon": 1000,
# }
#
# register_env(
#     "3DBall",
#     lambda c: Unity3DEnv(
#         file_name=env_config["file_name"],
#         no_graphics=False,
#         episode_horizon=env_config["episode_horizon"],
#     ),
# )
# ENV_NAME = "3DBall"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 1000.0



################

#ENV_NAME = "3DBall"