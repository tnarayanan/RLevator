import numpy as np


class StandardElevatorV5Controller:
    def __init__(self, env):
        self.env = env

    def predict(self, obs: np.array, deterministic=True):
        obs = obs.astype(np.int32)
        elevators = [{} for _ in range(self.env.num_elevators)]
        i = 2
        for elev in elevators:
            elev['floor'] = obs[i]
            elev['target'] = obs[i+1]
            elev['time_to_next_floor'] = obs[i+2]
            elev['prev_direction'] = obs[i+3]
            elev['direction'] = obs[i+4]
            i += 5
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

            if elev['direction'] == 0:
                # already moving down, select next closest floor that's below
                selected_action = max(down_floors_below)
            elif elev['direction'] == 2:
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

            # closest_stop_below = None
            # for i in reversed(range(0, elev['floor'])):
            #     if elev['requests'][i] == 1 or down_pressed[i] == 1:  # if person in elevator wants to get off at `i` or down is pressed on `i`
            #         closest_stop_below = i
            #         break
            #
            # closest_stop_above = None
            # for i in range(elev['floor'] + 1, self.env.num_floors):
            #     if elev['requests'][i] == 1 or up_pressed[i] == 1:  # if person in elevator wants to get off at `i` or up is pressed on `i`
            #         closest_stop_above = i
            #         break
            #
            # print(down_pressed, up_pressed)
            #
            # if elev['direction'] == 0 and closest_stop_below is not None:
            #     action[elev_id] = closest_stop_below
            # elif elev['direction'] == 2 and closest_stop_above is not None:
            #     action[elev_id] = closest_stop_above
            # elif closest_stop_below is not None:
            #     action[elev_id] = closest_stop_below
            # else:
            #     action[elev_id] = elev['floor'] if closest_stop_above is None else closest_stop_above

            action[elev_id] = selected_action

        return action, ""
