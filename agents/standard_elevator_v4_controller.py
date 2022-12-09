import numpy as np


class StandardElevatorV4Controller:
    def __init__(self, env):
        self.env = env

    def predict(self, obs: np.array, deterministic=True):
        obs = obs.astype(np.int32)

        num_floors_end = self.env.num_floors_end
        num_elevators_end = self.env.num_elevators_end

        i = 0
        # print(obs[i : i + num_elevators_end + 1])
        num_elevators = np.argmax(obs[i : i + num_elevators_end + 1])
        i += num_elevators_end + 1
        num_floors = np.argmax(obs[i : i + num_floors_end + 1])
        i += num_floors_end + 1
        elevators = [{} for _ in range(num_elevators)]
        for elev in elevators:
            elev['floor'] = np.argmax(obs[i : i + num_floors_end])
            i += num_floors_end
            elev['target'] = np.argmax(obs[i : i + num_floors_end])
            i += num_floors_end
            # elev['time_to_next_floor'] = obs[i+2]
            elev['direction'] = np.argmax(obs[i : i + 3])
            i += 3
            floor_requests = []
            for k in range(num_floors):
                floor_requests.append(obs[i+k])
            i += num_floors_end
            elev['requests'] = floor_requests

        up_pressed = []
        down_pressed = []
        for k in range(num_floors):
            up_pressed.append(obs[i+k])
        i += num_floors_end
        for k in range(num_floors):
            down_pressed.append(obs[i+k])

        # actual algo

        action = np.zeros(num_elevators_end * num_floors_end)

        for elev_id, elev in enumerate(elevators):
            closest_stop_below = None
            for i in reversed(range(0, elev['floor'])):
                if elev['requests'][i] == 1 or down_pressed[i] == 1:  # if person in elevator wants to get off at `i` or down is pressed on `i`
                    closest_stop_below = i
                    break

            closest_stop_above = None
            for i in range(elev['floor'] + 1, self.env.num_floors):
                if elev['requests'][i] == 1 or up_pressed[i] == 1:  # if person in elevator wants to get off at `i` or up is pressed on `i`
                    closest_stop_above = i
                    break

            cur_action = None

            if elev['direction'] == 0 and closest_stop_below is not None:
                cur_action = closest_stop_below
            elif elev['direction'] == 2 and closest_stop_above is not None:
                cur_action = closest_stop_above
            elif closest_stop_below is not None:
                cur_action = closest_stop_below
            else:
                cur_action = elev['floor'] if closest_stop_above is None else closest_stop_above

            # print(cur_action)
            action[elev_id * num_floors_end + cur_action] = 1

        return action, ""
