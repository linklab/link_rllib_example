import ray
import warnings

from tutorials.codes.single_agent.rllib_train import get_ray_config_and_ray_agent

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tutorials.codes.single_agent.rllib_algorithm import ALGORITHM
from tutorials.codes.single_agent.rllib_environment import ENV_NAME

if __name__ == "__main__":
	CHECKPOINT_PATH = "/Users/yhhan/ray_results/PPO_RandomWalk_2022-11-11_19-28-17g5z7t089/checkpoint_000005"

	ray_info = ray.init(local_mode=True, log_to_driver=True)

	ray_config, ray_agent = get_ray_config_and_ray_agent(algorithm=ALGORITHM, env_name=ENV_NAME)
	ray_agent.restore(checkpoint_path=CHECKPOINT_PATH)

	env = ray_agent.evaluation_workers.local_worker().env

	for epsiode in range(3):
		print("[[[ EPISODE: {0} ]]]".format(epsiode))
		state = env.reset()
		env.render(mode="human")
		done = False
		cumulative_reward = 0

		while not done:
			action = ray_agent.compute_single_action(state, explore=False)
			state, reward, done, _ = env.step(action)
			cumulative_reward += reward
			env.render(mode="human")

		print(cumulative_reward)
		print()

	ray.shutdown()
