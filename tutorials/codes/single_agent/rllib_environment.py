########## DISCRETE ACTION - ENVIRONMENTS ##########
# ENV_NAME = "CartPole-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 500
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = {}
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


# ENV_NAME = "Taxi-v3"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 10
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = {}
# CUSTOM_RAY_CONFIG_DDPG = None
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


# ENV_NAME = "PongDeterministic-v0"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 3
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = {}
# CUSTOM_RAY_CONFIG_DDPG = None
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


# ENV_NAME = "Acrobot-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 250
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = {}
# CUSTOM_RAY_CONFIG_DDPG = None
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


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
# CUSTOM_RAY_CONFIG_DQN = {}
# CUSTOM_RAY_CONFIG_DDPG = None
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


########## CONTINUOUS ACTION - ENVIRONMENTS ##########
# ENV_NAME = "MountainCarContinuous-v0"
# MAX_TRAIN_ITERATIONS = 200
# EPISODE_REWARD_AVG_SOLVED = 100.0
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = None
# CUSTOM_RAY_CONFIG_DDPG = {}
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


# ENV_NAME = "Pendulum-v1"
# MAX_TRAIN_ITERATIONS = 10000
# EPISODE_REWARD_AVG_SOLVED = -100
# NUM_EPISODES_EVALUATION = 5
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = None
# CUSTOM_RAY_CONFIG_DDPG = {}
# CUSTOM_RAY_CONFIG_PPO = {
# 	"train_batch_size": 512,
#     "vf_clip_param": 10.0,
#     "lambda": 0.1,
#     "gamma": 0.95,
#     "lr": 0.0003,
#     "sgd_minibatch_size": 64,
#     "num_sgd_iter": 6,
#     "grad_clip": 5.0
# }
# CUSTOM_RAY_CONFIG_SAC = {
#     "target_entropy": "auto",
#     "train_batch_size": 256,
#     "num_steps_sampled_before_learning_starts": 256,
#     "optimization" : {
#         "actor_learning_rate": 0.0003,
#         "critic_learning_rate": 0.0003,
#         "entropy_learning_rate": 0.0003
#     },
#     "grad_clip": 5.0,
# }


ENV_NAME = "BipedalWalker-v3"
MAX_TRAIN_ITERATIONS = 20000
EPISODE_REWARD_AVG_SOLVED = 250
NUM_EPISODES_EVALUATION = 3
ENV_CONFIG = {}
CUSTOM_RAY_CONFIG_DQN = None
CUSTOM_RAY_CONFIG_DDPG = {}
CUSTOM_RAY_CONFIG_PPO = {
    "train_batch_size": 2400,
    "vf_clip_param": 10.0,
    "lambda": 0.1,
    "gamma": 0.95,
    "lr": 0.0001,
    "sgd_minibatch_size": 1200,
    "num_sgd_iter": 10,
    "grad_clip": 5.0
}
CUSTOM_RAY_CONFIG_SAC = {
    "rollout_fragment_length": 1,
    "grad_clip": 5.0,
}


# ENV_NAME = "BipedalWalkerHardcore-v3"
# MAX_TRAIN_ITERATIONS = 3000
# EPISODE_REWARD_AVG_SOLVED = 250
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = None
# CUSTOM_RAY_CONFIG_DDPG = {}
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}


# ENV_NAME = "CarRacing-v1"
# MAX_TRAIN_ITERATIONS = 100
# EPISODE_REWARD_AVG_SOLVED = 100
# NUM_EPISODES_EVALUATION = 3
# ENV_CONFIG = {}
# CUSTOM_RAY_CONFIG_DQN = None
# CUSTOM_RAY_CONFIG_DDPG = {}
# CUSTOM_RAY_CONFIG_PPO = {}
# CUSTOM_RAY_CONFIG_SAC = {}
