import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import ray
import wandb
from datetime import datetime

from tutorials.codes.multi_agents.rllib_ma_utils import get_ray_config_and_ray_agent, print_ttt_iter_result, log_ttt_wandb
from tutorials.codes.multi_agents.rllib_ma_algorithm import ALGORITHM
from tutorials.codes.multi_agents.rllib_ma_environment import (
	ENV_NAME, MODE, ENV_CONFIG, MAX_TRAIN_ITERATIONS, EPISODE_REWARD_AVG_SOLVED
)

import gym


class RAY_RL:
	def __init__(
			self, env_name, algorithm, ray_config, ray_agent, max_train_iterations, episode_reward_avg_solved, use_wandb
	):
		self.env_name = env_name
		self.algorithm = algorithm
		self.ray_config = ray_config
		self.ray_agent = ray_agent

		self.max_train_iterations = max_train_iterations
		self.episode_reward_avg_solved = episode_reward_avg_solved
		self.use_wandb = use_wandb

		self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

		if self.use_wandb:
			self.wandb = wandb.init(
				project="{0}_{1}".format(self.algorithm, self.env_name),
				name=self.current_time,
				config=self.ray_config
			)

	def train_loop(self):
		num_optimizations_policy_O = 0
		num_optimizations_policy_X = 0

		for num_train in range(self.max_train_iterations):
			iter_result = self.ray_agent.train()

			# from pprint import pprint
			# pprint(iter_result)
			if "policy_O" in iter_result["info"]["learner"]:
				num_optimizations_policy_O += iter_result["info"]["learner"]["policy_O"]["num_agent_steps_trained"]

			if "policy_X" in iter_result["info"]["learner"]:
				num_optimizations_policy_X += iter_result["info"]["learner"]["policy_X"]["num_agent_steps_trained"]

			print_ttt_iter_result(iter_result, num_optimizations_policy_O, num_optimizations_policy_X)

			if self.use_wandb:
				log_ttt_wandb(self.wandb, iter_result, num_optimizations_policy_O, num_optimizations_policy_X)

			if MODE == 0:
				episode_reward_mean = iter_result["evaluation"]["policy_reward_mean"]["policy_O"]
			elif MODE == 1:
				episode_reward_mean = iter_result["evaluation"]["policy_reward_mean"]["policy_X"]
			elif MODE == 2 and MODE == 3:
				episode_reward_mean = 10
			else:
				raise ValueError()

			if episode_reward_mean >= self.episode_reward_avg_solved:
				checkpoint_path = ray_agent.save()
				print("*** Solved with Evaluation Episodes Reward Mean: {0:>6.2f} ({1} Evaluation Episodes).".format(
					episode_reward_mean, iter_result["evaluation"]["episodes_this_iter"]
				))
				print("*** Checkpoint at {0}".format(checkpoint_path))
				break


if __name__ == "__main__":
	print("RAY VERSION: {0}".format(ray.__version__))
	print("GYM VERSION: {0}".format(gym.__version__))

	ray_info = ray.init(local_mode=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(
		algorithm=ALGORITHM, env_name=ENV_NAME, env_config=ENV_CONFIG, num_workers=1
	)

	if MODE == 0:
		ray_config.policies_to_train = ["policy_O"]

	print("#" * 128)
	print(ray_agent.get_policy(policy_id="policy_O").model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_O").observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_O").action_space)))

	print("#" * 128)
	print(ray_agent.get_policy(policy_id="policy_X").model)
	print("OBSERVATION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_X").observation_space)))
	print("ACTION SPACE: {0}".format(str(ray_agent.get_policy(policy_id="policy_X").action_space)))

	ray_rl = RAY_RL(
		env_name=ENV_NAME, algorithm=ALGORITHM, ray_config=ray_config, ray_agent=ray_agent,
		max_train_iterations=MAX_TRAIN_ITERATIONS, episode_reward_avg_solved=EPISODE_REWARD_AVG_SOLVED,
		use_wandb=True
	)

	ray_rl.train_loop()

	ray.shutdown()
