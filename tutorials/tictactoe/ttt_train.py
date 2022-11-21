import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import random

from tutorials.tictactoe.tic_tac_toe_env import TicTacToe

if __name__ == "__main__":
    ray.init()
    env = TicTacToe()
    num_policies = 4
    policies = {
        "policy_{}".format(i): (None, env.observation_space, env.action_space, {})
        for i in range(num_policies)
    }
    policy_ids = list(policies.keys())

    config = {
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda agent_id: random.choice(policy_ids)),
        },
        "framework": "torch",
        "num_workers": 2  # Adjust this according to the number of CPUs on your machine.
    }

    print(config)

    trainer = PPOTrainer(env=TicTacToe, config=config)
    best_eps_len = 0
    mean_reward_threshold = -1

    while True:
        results = trainer.train()
        #print(pretty_print(results))

        if results["episode_reward_mean"] > mean_reward_threshold and results["episode_len_mean"] > best_eps_len:
            trainer.save("ttt_model")
            best_eps_len = results["episode_len_mean"]
            print("--------------------- MODEL SAVED!")

        if results.get("timesteps_total") > 10 ** 7:
            break

    ray.shutdown()