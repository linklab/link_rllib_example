import time

import ray
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tutorials.codes.multi_agents.rllib_ma_utils import get_ray_config_and_ray_agent
from tutorials.codes.multi_agents.rllib_ma_algorithm import ALGORITHM
from tutorials.codes.multi_agents.rllib_ma_environment import ENV_NAME, ENV_CONFIG

if __name__ == "__main__":
	CHECKPOINT_PATH = "/Users/yhhan/ray_results/PPO_RandomWalk_2022-11-22_23-13-43541kn6x6/checkpoint_000005"

	ray_info = ray.init(local_mode=True, log_to_driver=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(algorithm=ALGORITHM, env_name=ENV_NAME, env_config=ENV_CONFIG)
	ray_agent.restore(checkpoint_path=CHECKPOINT_PATH)

	env = ray_agent.evaluation_workers.local_worker().env

	for epsiode in range(3):
		print("[[[ EPISODE: {0} ]]]".format(epsiode))

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

			if 'O' in rewards:
				cumulative_reward['O'] += rewards['O']
			if 'X' in rewards:
				cumulative_reward['X'] += rewards['X']

			env.render()

			if policy_id == "policy_O":
				policy_id = "policy_X"
			elif policy_id == "policy_X":
				policy_id = "policy_O"
			else:
				raise ValueError()

			observations = next_observations
			time.sleep(0.1)

		print(cumulative_reward)
		print()

	ray.shutdown()
