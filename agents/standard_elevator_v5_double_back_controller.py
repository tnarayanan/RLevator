import numpy as np


class StandardElevatorV5GoBackController:
    def __init__(self, env):
        self.env = env

    def predict(self, obs: np.array, deterministic=True):
        obs = obs.astype(np.int32)
        elevators = [{} for _ in range(self.env.num_elevators)]
        i = 2
        for elev in elevators:
            elev['floor'] = obs[i]
            elev['target'] = obs[i+1]
            elev['prev_direction'] = obs[i+2]
            elev['direction'] = obs[i+3]
            i += 4
            floor_requests = []
            for k in range(self.env.num_floors):
                floor_requests.append(obs[i+k])
            i += self.env.num_floors_end
            elev['requests'] = floor_requests

        up_pressed = []
        down_pressed = []
        for k in range(self.env.num_floors):
            up_pressed.append(obs[i+k])
        i += self.env.num_floors_end
        for k in range(self.env.num_floors):
            down_pressed.append(obs[i+k])

        # actual algo

        action = np.zeros(self.env.num_elevators_end)

        for elev_id, elev in enumerate(elevators):
            all_stops_down = set()
            all_stops_up = set()
            for i in range(self.env.num_floors):
                if elev['requests'][i] == 1:
                    if i > elev['floor']:
                        all_stops_up.add(i)
                    else:
                        all_stops_down.add(i)
                if down_pressed[i]:
                    all_stops_down.add(i)
                if up_pressed[i]:
                    all_stops_up.add(i)

            selected_action = None

            down_floors_below = [floor for floor in all_stops_down if floor < elev['floor']]
            down_floors_above = [floor for floor in all_stops_down if floor > elev['floor']]
            up_floors_below = [floor for floor in all_stops_up if floor < elev['floor']]
            up_floors_above = [floor for floor in all_stops_down if floor > elev['floor']]

            if elev['direction'] == 0 and len(down_floors_below) > 0:
                # already moving down, select next closest floor that's below
                selected_action = max(down_floors_below)
            elif elev['direction'] == 2 and len(up_floors_above) > 0:
                selected_action = min(up_floors_above)
            else:
                if elev['prev_direction'] == 0:
                    # was moving down before, check below first
                    if len(down_floors_below) > 0:
                        selected_action = max(down_floors_below)
                    elif len(up_floors_below) > 0:
                        selected_action = min(up_floors_below)
                    elif len(up_floors_above) > 0:
                        selected_action = min(up_floors_above)
                    elif len(down_floors_above) > 0:
                        selected_action = max(down_floors_above)
                    else:
                        selected_action = elev['floor']
                else:
                    # was moving up before, check above first
                    if len(up_floors_above) > 0:
                        selected_action = min(up_floors_above)
                    elif len(down_floors_above) > 0:
                        selected_action = max(down_floors_above)
                    elif len(down_floors_below) > 0:
                        selected_action = max(down_floors_below)
                    elif len(up_floors_below) > 0:
                        selected_action = min(up_floors_below)
                    else:
                        selected_action = elev['floor']

            action[elev_id] = selected_action

        return action, ""
