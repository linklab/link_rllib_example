########## DISCRETE ACTION - ENVIRONMENTS ##########
# ENV_NAME = "CartPole-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 500
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "Taxi-v3"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 10
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "PongDeterministic-v0"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 3
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "Acrobot-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 250
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# from ray.tune import register_env
# from tutorials.codes.custom_env.random_walk import RandomWalk
# register_env("RandomWalk", lambda env_config: RandomWalk(env_config))
# ENV_NAME = "RandomWalk"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 1.0
# NUM_EPISODES_EVALUATION = 3
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
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "Pendulum-v1"
# MAX_TRAIN_ITERATIONS = 10000
# EPISODE_REWARD_AVG_SOLVED = -100
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {
# 	"train_batch_size": 512,
#     "vf_clip_param": 10.0,
#     "lambda": 0.1,
#     "gamma": 0.95,
#     "lr": 0.0003,
#     "sgd_minibatch_size": 64,
#     "num_sgd_iter": 6,
#     "observation_filter": "MeanStdFilter"
# }


ENV_NAME = "BipedalWalker-v3"
MAX_TRAIN_ITERATIONS = 1000
EPISODE_REWARD_AVG_SOLVED = 250
NUM_EPISODES_EVALUATION = 3
ENV_CONFIG = {}
CUSTOM_RAY_CONFIG = {
    "train_batch_size": 512,
    "vf_clip_param": 10.0,
    "lambda": 0.1,
    "gamma": 0.95,
    "lr": 0.0003,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 6,
    "observation_filter": "MeanStdFilter"
}


# ENV_NAME = "BipedalWalkerHardcore-v3"
# MAX_TRAIN_ITERATIONS = 3000
# EPISODE_REWARD_AVG_SOLVED = 250
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}


# ENV_NAME = "CarRacing-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 100
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG = {}
