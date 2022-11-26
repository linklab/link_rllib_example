import time

import ray
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tutorials.codes.multi_agents.rllib_ttt_utils import get_ttt_ray_config_and_ray_agent
from tutorials.codes.multi_agents.rllib_ttt_algorithm import ALGORITHM
from tutorials.codes.multi_agents.rllib_ttt_environment import ENV_NAME, ENV_CONFIG, CUSTOM_RAY_CONFIG

if __name__ == "__main__":
	ray_info = ray.init(local_mode=True)

	ray_config, ray_agent = get_ttt_ray_config_and_ray_agent(
		algorithm=ALGORITHM, env_name=ENV_NAME, env_config=ENV_CONFIG, custom_ray_config=CUSTOM_RAY_CONFIG
	)
	ray_agent.restore(
		checkpoint_path="/Users/yhhan/ray_results/PPO_TicTacToe343_2022-11-26_20-44-29dqn0xk08/checkpoint_000043"
	)

	env = ray_agent.evaluation_workers.local_worker().env

	for epsiode in range(3):
		print("[[[ EPISODE: {0} ]]]".format(epsiode))

		print("RESET!")
		observations = env.reset()

		env.render()

		dones = dict()
		dones["__all__"] = False

		cumulative_reward = {
			'O': 0.0,
			'X': 0.0
		}

		policy_id = 'policy_O'

		while not dones["__all__"]:
			actions = ray_agent.compute_actions(observations, policy_id=policy_id, explore=False)

			next_observations, rewards, dones, infos = env.step(actions)

			print(
				"Obs.: {0}, Action: {1}, Next Obs.: {2}, Reward: {3}, Done: {4}, Info: {5}".format(
					observations, actions, next_observations, rewards, dones, infos
			))

			if 'O' in rewards:
				cumulative_reward['O'] += rewards['O']

			if 'X' in rewards:
				cumulative_reward['X'] += rewards['X']

			env.render()

			policy_id = "policy_X" if policy_id == "policy_O" else "policy_O"

			observations = next_observations
			time.sleep(0.1)

		print("CUMULATIVE REWARD: {0}".format(cumulative_reward))
		print(end="\n" * 5)

	ray.shutdown()
