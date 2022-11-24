from gym import spaces
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec


def get_ray_config_and_ray_agent(algorithm_policy_o, algorithm_policy_x, env_name, env_config, num_workers=1):
    if algorithm_policy_o == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig
        from ray.rllib.algorithms.dqn import DQNTorchPolicy
        ray_config_policy_o = DQNConfig()
        ray_policy_policy_o = DQNTorchPolicy
    elif algorithm_policy_o == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.ppo import PPOTorchPolicy
        ray_config_policy_o = PPOConfig()
        ray_policy_policy_o = PPOTorchPolicy
    elif algorithm_policy_o == "SAC":
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.sac import SACTorchPolicy
        ray_config_policy_o = SACConfig()
        ray_policy_policy_o = SACTorchPolicy
    elif algorithm_policy_o == "Dummy":
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.sac import SACTorchPolicy
        ray_config_policy_o = SACConfig()
        ray_policy_policy_o = SACTorchPolicy
    else:
        raise ValueError()

    if algorithm_policy_x == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig
        from ray.rllib.algorithms.dqn import DQNTorchPolicy
        ray_config_policy_x = DQNConfig()
        ray_policy_policy_x = DQNTorchPolicy
    elif algorithm_policy_x == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.algorithms.ppo import PPOTorchPolicy
        ray_config_policy_x = PPOConfig()
        ray_policy_policy_x = PPOTorchPolicy
    elif algorithm_policy_x == "SAC":
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.sac import SACTorchPolicy
        ray_config_policy_x = SACConfig()
        ray_policy_policy_x = SACTorchPolicy
    elif algorithm_policy_x == "Dummy":
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.algorithms.sac import SACTorchPolicy
        ray_config_policy_x = SACConfig()
        ray_policy_policy_x = SACTorchPolicy
    else:
        raise ValueError()

    ray_config_policy_o.framework_str = "torch"
    ray_config_policy_x.framework_str = "torch"

    ray_config = AlgorithmConfig()

    ray_config.env = env_name
    ray_config.env_config = env_config
    ray_config.framework_str = "torch"
    ray_config.num_workers = num_workers
    ray_config.evaluation_interval = 1  # 평가를 위한 훈련 간격
    ray_config.evaluation_duration = 5  # 평가를 위한 에피소드 개수

    policies = {
        "policy_O": PolicySpec(
            config=ray_config_policy_o.to_dict(),
            policy_class=ray_policy_policy_o,
            observation_space=spaces.Box(0.0, 2.0, (12,), float),
            action_space=spaces.Discrete(n=12)
        ),
        "policy_X": PolicySpec(
            config=ray_config_policy_x.to_dict(),
            policy_class=ray_policy_policy_x,
            observation_space=spaces.Box(0.0, 2.0, (12,), float),
            action_space=spaces.Discrete(n=12)
        )
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "O":
            return "policy_O"
        elif agent_id == "X":
            return "policy_X"
        else:
            raise ValueError()

    ray_config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    ray_config.disable_env_checking = True

    from ray.rllib.algorithms import Algorithm
    ray_agent = Algorithm(config=ray_config)


    # if env_name == "3DBall":
    #     policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(env_name)
    # elif env_name == "Kart":
    #     policies = {
    #         "Kart": PolicySpec(
    #             observation_space=spaces.Box(low=-10_000, high=10_000, shape=(48,)),
    #             action_space=spaces.Box(low=-10_000, high=10_000, shape=(1,)),
    #         ),
    #     }
    #     def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #         return "Kart"
    # elif env_name == "TicTacToe343":
    #     policies = {
    #         "policy_O": PolicySpec(
    #             observation_space=spaces.Box(0.0, 2.0, (12,), float),
    #             action_space=spaces.Discrete(n=12)
    #         ),
    #         "policy_X": PolicySpec(
    #             observation_space=spaces.Box(0.0, 2.0, (12,), float),
    #             action_space=spaces.Discrete(n=12)
    #         ),
    #     }
    #     def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    #         if agent_id == "O":
    #             return "policy_O"
    #         elif agent_id == "X":
    #             return "policy_X"
    #         else:
    #             raise ValueError()
    # else:
    #     raise ValueError()
    #
    # ray_config.multi_agent(
    #     policies=policies, policy_mapping_fn=policy_mapping_fn
    # )

    # ray_config.rollouts(
    #     num_rollout_workers=0,
    #     no_done_at_end=True,
    #     rollout_fragment_length=200,
    # )
    # ray_config.training(
    #     lr=0.0003,
    #     lambda_=0.95,
    #     gamma=0.99,
    #     sgd_minibatch_size=256,
    #     train_batch_size=4000,
    #     num_sgd_iter=20,
    #     clip_param=0.2,
    #     model={"fcnet_hiddens": [512, 512]}
    # )
    # ray_config.resources(num_gpus=int("0"))
    #
    # ray_config.environment(
    #     "3DBall",
    #     env_config={
    #         "file_name": "/Users/zero/PycharmProjects/link_rllib_example/unity_env/3DBall_Darwin",
    #         "episode_horizon": 1000,
    #     },
    #     disable_env_checking=True
    # )

    # if algorithm == "DQN":
    #     from ray.rllib.algorithms.dqn import DQN
    #     ray_agent = DQN(ray_config, env_name)
    # elif algorithm == "PPO":
    #     from ray.rllib.algorithms.ppo import PPO
    #     ray_agent = PPO(ray_config, env_name)
    # elif algorithm == "DDPG":
    #     from ray.rllib.algorithms.ddpg import DDPG
    #     ray_agent = DDPG(ray_config, env_name)
    # elif algorithm == "SAC":
    #     from ray.rllib.algorithms.sac import SAC
    #     ray_agent = SAC(ray_config, env_name)
    # else:
    #     raise ValueError()

    return ray_config, ray_agent


def print_ttt_iter_result(iter_result, num_optimizations_policy_O, num_optimizations_policy_X):
    prefix = "{0:>2} | episodes: {1:>3} | timesteps: {2:>7,} | opts (policy_O).: {3:>6,d} | opts (policy_X).: {4:>6,d}".format(
        iter_result["training_iteration"], iter_result["episodes_total"], iter_result["timesteps_total"],
        int(num_optimizations_policy_O), int(num_optimizations_policy_X)
    )

    episode_reward_policy_O = "epi_reward_mean: {0:>6.2f}".format(
        iter_result["sampler_results"]["policy_reward_mean"]["policy_O"]
    )

    episode_reward_policy_X = "epi_reward_mean: {0:>6.2f}".format(
        iter_result["sampler_results"]["policy_reward_mean"]["policy_X"]
    )

    evaluation_episode_reward_policy_O = "eval_epi_reward_mean: {0:>6.2f}".format(
        iter_result["evaluation"]["policy_reward_mean"]["policy_O"]
    )

    evaluation_episode_reward_policy_X = "eval_epi_reward_mean: {0:>6.2f}".format(
        iter_result["evaluation"]["policy_reward_mean"]["policy_X"]
    )

    loss_policy_O = "avg. loss: N/A"
    if "policy_O" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_O"]:
            loss_policy_O = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_O"]["mean_td_error"]
            )
        else:
            loss_policy_O = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_O"]["learner_stats"]["total_loss"]
            )

    loss_policy_X = "avg. loss: N/A"
    if "policy_X" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_X"]:
            loss_policy_X = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_X"]["mean_td_error"]
            )
        else:
            loss_policy_X = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["policy_X"]["learner_stats"]["total_loss"]
            )


    time = "time: {0:>6.2f} sec.".format(
        iter_result["time_total_s"]
    )

    print("[{0}] [policy_O: {1}, {2}, {3}] [policy_X:{4}, {5}, {6}] {7}".format(
        prefix,
        episode_reward_policy_O, evaluation_episode_reward_policy_O, loss_policy_O,
        episode_reward_policy_X, evaluation_episode_reward_policy_X, loss_policy_X,
        time
    ))


def log_ttt_wandb(wandb, iter_result, num_optimizations_policy_O, num_optimizations_policy_X):
    log_dict = {
        "train": iter_result["training_iteration"],
        "episodes": iter_result["episodes_total"],
        "timesteps": iter_result["timesteps_total"],
        "policy_O/optimizations": num_optimizations_policy_O,
        "policy_O/episode_reward_mean": iter_result["sampler_results"]["policy_reward_mean"]["policy_O"],
        "policy_O/evaluation_episode_reward_mean": iter_result["evaluation"]["policy_reward_mean"]["policy_O"],
        "policy_X/optimizations": num_optimizations_policy_X,
        "policy_X/episode_reward_mean": iter_result["sampler_results"]["policy_reward_mean"]["policy_X"],
        "policy_X/evaluation_episode_reward_mean": iter_result["evaluation"]["policy_reward_mean"]["policy_X"],
    }

    if "policy_O" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_O"]:
            log_dict["policy_O/loss"] = iter_result["info"]["learner"]["policy_O"]["mean_td_error"]
        else:
            log_dict["policy_O/loss"] = iter_result["info"]["learner"]["policy_O"]["learner_stats"]["total_loss"]

    if "policy_X" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["policy_X"]:
            log_dict["policy_X/loss"] = iter_result["info"]["learner"]["policy_X"]["mean_td_error"]
        else:
            log_dict["policy_X/loss"] = iter_result["info"]["learner"]["policy_X"]["learner_stats"]["total_loss"]

    wandb.log(log_dict)

