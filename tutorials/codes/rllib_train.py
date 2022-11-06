import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ray
import wandb
from datetime import datetime
from tutorials.codes.rllib_utils import get_ray_config_and_ray_agent, print_iter_result, log_wandb


class RAY_RL:
	def __init__(self, env_name, ray_config, ray_agent, max_train_iterations, episode_reward_avg_solved, use_wandb):
		self.env_name = env_name
		self.ray_config = ray_config
		self.ray_agent = ray_agent

		self.max_train_iterations = max_train_iterations
		self.episode_reward_avg_solved = episode_reward_avg_solved
		self.use_wandb = use_wandb

		self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

		if self.use_wandb:
			self.wandb = wandb.init(
				project="DQN_{0}".format(self.env_name),
				name=self.current_time,
				config=ray_config
			)

	def train_loop(self):
		optimizations = 0

		for num_train in range(self.max_train_iterations):
			iter_result = self.ray_agent.train()

			# from ray.tune.logger import pretty_print
			# print(pretty_print(iter_result))

			if "default_policy" in iter_result["info"]["learner"]:
				optimizations += iter_result["info"]["learner"]["default_policy"]["num_agent_steps_trained"]
			else:
				optimizations += 0

			print_iter_result(iter_result, optimizations)

			if self.use_wandb:
				log_wandb(iter_result, optimizations)

			episode_reward_mean = iter_result["evaluation"]["episode_reward_mean"]

			if episode_reward_mean >= self.episode_reward_avg_solved:
				checkpoint_path = ray_agent.save()
				print("*** Solved with Evaluation Episodes Reward Mean: {0:>6.2f} ({1} Evaluation Episodes).".format(
					iter_result["evaluation"]["episode_reward_mean"],
					iter_result["evaluation"]["episodes_this_iter"]
				))
				print("*** Checkpoint at {0}".format(checkpoint_path))
				break


if __name__ == "__main__":
	# ENV_NAME = "CartPole-v1"
	# MAX_TRAIN_ITERATIONS = 100
	# EPISODE_REWARD_AVG_SOLVED = 500

	# ENV_NAME = "Taxi-v3"
	# MAX_TRAIN_ITERATIONS = 100
	# EPISODE_REWARD_AVG_SOLVED = 10

	# ENV_NAME = "PongDeterministic-v0"
	# MAX_TRAIN_ITERATIONS = 100
	# EPISODE_REWARD_AVG_SOLVED = 3

	ENV_NAME = "MountainCarContinuous-v0"
	MAX_TRAIN_ITERATIONS = 200
	EPISODE_REWARD_AVG_SOLVED = 100.0

	# ALGORITHM = "DQN"
	ALGORITHM = "PPO"
	# ALGORITHM = "SAC"

	ray_info = ray.init(log_to_driver=False)

	ray_config, ray_agent = get_ray_config_and_ray_agent(algorithm=ALGORITHM, env_name=ENV_NAME, num_workers=3)

	print(ray_agent.get_policy().model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy().observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy().action_space)))

	ray_rl = RAY_RL(
		env_name=ENV_NAME, ray_config=ray_config, ray_agent=ray_agent,
		max_train_iterations=MAX_TRAIN_ITERATIONS, episode_reward_avg_solved=EPISODE_REWARD_AVG_SOLVED,
		use_wandb=False
	)

	ray_rl.train_loop()

	ray.shutdown()
