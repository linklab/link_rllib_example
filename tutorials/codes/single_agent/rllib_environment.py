########## DISCRETE ACTION - ENVIRONMENTS ##########
# ENV_NAME = "CartPole-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 500
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "Taxi-v3"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 10
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "PongDeterministic-v0"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "Acrobot-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 250
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# from ray.tune import register_env
# from tutorials.codes.custom_env.random_walk import RandomWalk
# register_env("RandomWalk", lambda env_config: RandomWalk(env_config))
# ENV_NAME = "RandomWalk"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 1.0
# ENV_CONFIG = {
#     "num_internal_states": 5,       # 종료 상태를 제외한 내부 상태 개수
#     "transition_reward": 0.0,       # 일반적인 상태 전이 보상
#     "left_terminal_reward": 0.0,    # 왼쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
#     "right_terminal_reward": 1.0    # 오른쪽 종료 상태로 이동하는 행동 수행 시 받는 보상
# }
# CUSTOM_RAY_CONFIG = {}


########## CONTINUOUS ACTION - ENVIRONMENTS ##########
# ENV_NAME = "MountainCarContinuous-v0"
# MAX_TRAIN_ITERATIONS = 200
# EPISODE_REWARD_AVG_SOLVED = 100.0
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


ENV_NAME = "Pendulum-v1"
MAX_TRAIN_ITERATIONS = 10000
EPISODE_REWARD_AVG_SOLVED = 0
ENV_CONFIG = {}
CUSTOM_RAY_CONFIG = {
	"train_batch_size": 512,
    "vf_clip_param": 10.0,
    "num_workers": 0,
    "num_envs_per_worker": 20,
    "lambda": 0.1,
    "gamma": 0.95,
    "lr": 0.0003,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 6,
    "observation_filter": "MeanStdFilter"
}


# ENV_NAME = "BipedalWalker-v3"
# MAX_TRAIN_ITERATIONS = 1000
# EPISODE_REWARD_AVG_SOLVED = 250
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {
#     "lr": 0.0001,
#     "num_sgd_iter": 50,
#     "sgd_minibatch_size": 250,
#     "fcnet_hiddens": [512, 512]
# }


# ENV_NAME = "BipedalWalkerHardcore-v3"
# MAX_TRAIN_ITERATIONS = 3000
# EPISODE_REWARD_AVG_SOLVED = 250
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "CarRacing-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 100
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# from ray.rllib.env import Unity3DEnv
# register_env(
# 	"Kart_Darwin",
# 	lambda env_config: Unity3DEnv(
# 		file_name=env_config["file_name"],
# 		no_graphics=env_config["no_graphics"],
# 		episode_horizon=env_config["episode_horizon"],
# 	)
# )
# ENV_NAME = "Kart_Darwin"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 1.0
# ENV_CONFIG = {
# 	"file_name": "/Users/zero/PycharmProjects/link_rllib_example/unity_env/Kart_Darwin",
#   "no_graphics": False
# 	"episode_horizon": 1000,
# }
# CUSTOM_RAY_CONFIG = {}