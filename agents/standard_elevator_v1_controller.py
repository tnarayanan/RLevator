import numpy as np


class StandardElevatorV1Controller:
    def __init__(self, env):
        self.env = env

    def predict(self, obs: np.array, deterministic=True):
        obs = obs.astype(np.int32)
        elevators = [{} for _ in range(self.env.num_elevators)]
        i = 0
        for elev in elevators:
            elev['floor'] = obs[i]
            elev['target'] = obs[i+1]
            elev['time_to_next_floor'] = obs[i+2]
            elev['direction'] = obs[i+3]
            i += 4
            floor_requests = []
            for k in range(self.env.num_floors):
                floor_requests.append(obs[i+k])
            i += self.env.num_floors
            elev['requests'] = floor_requests

        pending_requests = []
        for k in range(self.env.num_floors):
            pending_requests.append(obs[i+k])

        # actual algo

        action = np.zeros(self.env.num_elevators)

        for elev_id, elev in enumerate(elevators):
            closest_request_below = elev['floor']
            for i in reversed(range(0, elev['floor'])):
                if elev['requests'][i] > 0 or pending_requests[i] > 0:
                    closest_request_below = i
                    break

            closest_request_above = elev['floor']
            for i in range(elev['floor'] + 1, self.env.num_floors):
                if elev['requests'][i] > 0 or pending_requests[i] > 0:
                    closest_request_above = i
                    break

            if elev['direction'] == 0 and closest_request_below != elev['floor']:
                action[elev_id] = closest_request_below
            elif elev['direction'] == 2 and closest_request_above != elev['floor']:
                action[elev_id] = closest_request_above
            elif closest_request_below != elev['floor']:
                action[elev_id] = closest_request_below
            else:
                action[elev_id] = closest_request_above

        return action, ""
