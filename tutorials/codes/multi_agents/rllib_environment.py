########## DISCRETE ACTION - ENVIRONMENTS ##########
from ray.rllib.env import Unity3DEnv
from ray.tune.registry import register_env
from tutorials.codes.custom_env.tic_tac_toe_343 import TicTacToe343

# env_config = {
# 	"num_internal_states": 5,  # 종료 상태를 제외한 내부 상태 개수
# 	"transition_reward": 0.0,  # 일반적인 상태 전이 보상
# 	"left_terminal_reward": 0.0,  # 왼쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
# 	"right_terminal_reward": 1.0  # 오른쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
# }
# register_env("TicTacToe343", lambda config: TicTacToe343(env_config))
# ENV_NAME = "TicTacToe343"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 1.0


################
env_name = "/Users/zero/PycharmProjects/link_rllib_example/unity_env/3DBall_Darwin"
env_config={
    "file_name": env_name,
    "episode_horizon": 1000,
}

register_env(
    "3DBall",
    lambda c: Unity3DEnv(
        file_name=env_config["file_name"],
        no_graphics=False,
        episode_horizon=env_config["episode_horizon"],
    ),
)
ENV_NAME = "3DBall"
MAX_TRAIN_ITERATIONS = 100
EPISODE_REWARD_AVG_SOLVED = 1000.0



################

#ENV_NAME = "3DBall"