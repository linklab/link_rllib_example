def get_ray_config_and_ray_agent(algorithm, env_name, env_config, num_workers=1):
    if algorithm == "DQN":
        from ray.rllib.algorithms.dqn import DQNConfig
        ray_config = DQNConfig()
    elif algorithm == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig
        ray_config = PPOConfig()
    elif algorithm == "DDPG":
        from ray.rllib.algorithms.ddpg import DDPGConfig
        ray_config = DDPGConfig()
    elif algorithm == "SAC":
        from ray.rllib.algorithms.sac import SACConfig
        ray_config = SACConfig()
    else:
        raise ValueError()

    ray_config.framework_str = "torch"
    ray_config.num_workers = num_workers
    ray_config.evaluation_interval = 1  # 평가를 위한 훈련 간격
    ray_config.evaluation_duration = 5  # 평가를 위한 에피소드 개수
    ray_config.env_config = env_config

    if algorithm == "DQN":
        from ray.rllib.algorithms.dqn import DQN
        ray_agent = DQN(ray_config, env_name)
    elif algorithm == "PPO":
        from ray.rllib.algorithms.ppo import PPO
        ray_agent = PPO(ray_config, env_name)
    elif algorithm == "DDPG":
        from ray.rllib.algorithms.ddpg import DDPG
        ray_agent = DDPG(ray_config, env_name)
    elif algorithm == "SAC":
        from ray.rllib.algorithms.sac import SAC
        ray_agent = SAC(ray_config, env_name)
    else:
        raise ValueError()

    return ray_config, ray_agent


def print_iter_result(iter_result, optimizations):
    prefix = "{0:>2} | episodes: {1:>3} | timesteps: {2:>7,} | opts.: {3:>6,d}".format(
        iter_result["training_iteration"], iter_result["episodes_total"],
        iter_result["timesteps_total"], int(optimizations)
    )

    episode_reward = "epi_reward(mean/min/max): {0:>6.2f}/{1:>6.2f}/{2:>6.2f}".format(
        iter_result["episode_reward_mean"], iter_result["episode_reward_min"],
        iter_result["episode_reward_max"]
    )

    evaluation_episode_reward = "eval_epi_reward(mean/min/max): {0:>6.2f}/{1:>6.2f}/{2:>6.2f}".format(
        iter_result["evaluation"]["episode_reward_mean"], iter_result["evaluation"]["episode_reward_min"],
        iter_result["evaluation"]["episode_reward_max"]
    )

    if "default_policy" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["default_policy"]:
            loss = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["default_policy"]["mean_td_error"]
            )
        else:
            loss = "avg. loss: {0:>6.3f}".format(
                iter_result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]
            )
    else:
        loss = "avg. loss: N/A"

    time = "time: {0:>6.2f} sec.".format(
        iter_result["time_total_s"]
    )

    print("[{0}] {1}, {2}, {3}, {4}".format(prefix, episode_reward, evaluation_episode_reward, loss, time))


def log_wandb(wandb, iter_result, optimizations):
    log_dict = {
        "train": iter_result["training_iteration"],
        "episodes": iter_result["episodes_total"],
        "timesteps": iter_result["timesteps_total"],
        "optimizations": int(optimizations),
        "train/episode_reward_mean": iter_result["episode_reward_mean"],
        "train/episode_reward_min": iter_result["episode_reward_min"],
        "train/episode_reward_max": iter_result["episode_reward_max"],
        "train/episode_length_mean": iter_result["episode_len_mean"],
        "evaluation/episode_reward_mean": iter_result["evaluation"]["episode_reward_mean"],
        "evaluation/episode_reward_min": iter_result["evaluation"]["episode_reward_min"],
        "evaluation/episode_reward_max": iter_result["evaluation"]["episode_reward_max"],
        "evaluation/episode_length_mean": iter_result["evaluation"]["episode_len_mean"],
    }
    if "default_policy" in iter_result["info"]["learner"]:
        if 'mean_td_error' in iter_result["info"]["learner"]["default_policy"]:
            log_dict["loss"] = iter_result["info"]["learner"]["default_policy"]["mean_td_error"]
        else:
            log_dict["loss"] = iter_result["info"]["learner"]["default_policy"]["learner_stats"]["total_loss"]

    wandb.log(log_dict)

