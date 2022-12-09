import numpy as np


class StandardElevatorV3Controller:
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
            # all_stops_down = set()
            # all_stops_up = set()
            # for i in range(self.env.num_floors):
            #     if elev['requests'][i] == 1:
            #         if i > elev['floor']:
            #             all_stops_up.add(i)
            #         else:
            #             all_stops_down.add(i)
            #     if down_pressed[i]:
            #         all_stops_down.add(i)
            #     if up_pressed[i]:
            #         all_stops_up.add(i)
            #
            # selected_action = None
            # if elev['direction'] == 0:
            #     # already moving down, select next closest floor that's below
            #     selected_action = max([floor for floor in all_stops_down if floor < elev['floor']])
            # elif elev['direction'] == 2:
            #     selected_action = min([floor for floor in all_stops_down if floor > elev['floor']])
            # else:
            #

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

            print(down_pressed, up_pressed)

            if elev['direction'] == 0 and closest_stop_below is not None:
                action[elev_id] = closest_stop_below
            elif elev['direction'] == 2 and closest_stop_above is not None:
                action[elev_id] = closest_stop_above
            elif closest_stop_below is not None:
                action[elev_id] = closest_stop_below
            else:
                action[elev_id] = elev['floor'] if closest_stop_above is None else closest_stop_above

        return action, ""
