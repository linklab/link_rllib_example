import random


class Dummy_Agents:
    def __init__(self):
        pass

    def get_actions(self, observations):
        agent = list(observations.keys())[0]
        available_action_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        import random
        action = {
            agent: random.choice(available_action_ids)
        }

        return action
