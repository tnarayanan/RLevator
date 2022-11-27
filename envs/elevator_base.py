from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class Request:
    # TODO: add some unique field here so that two identical requests don't get hashed to the same thing
    time_requested: int
    target_floor: int


class ElevatorState(IntEnum):
    MOVING_DOWN = 0
    IDLE = 1
    MOVING_UP = 2


TIME_PER_FLOOR = 5


class Elevator:
    def __init__(self, floor: int, target_floor: int, num_floors: int):
        self.floor: int = floor
        self.target_floor: int = target_floor
        self.time_to_next_floor: int = 0
        self.state: ElevatorState = ElevatorState.IDLE
        self.requests: dict[int, set[Request]] = {}

        self.num_floors = num_floors

    def add_request(self, request: Request):
        if request.target_floor not in self.requests:
            self.requests[request.target_floor] = set()
        self.requests[request.target_floor].add(request)

    def batch_add_requests(self, requests: set[Request], floor: int | None = None):
        if floor is None:
            floor = self.floor
        if floor not in self.requests:
            self.requests[floor] = set()
        self.requests[floor].update(requests)

    def remove_request(self, request: Request):
        if request not in self.requests[request.target_floor]:
            return
        self.requests[request.target_floor].remove(request)

    def batch_remove_requests(self, floor: int | None = None) -> int:
        if floor is None:
            floor = self.floor
        if floor not in self.requests:
            return 0
        num_requests = len(self.requests[self.floor])
        self.requests[self.floor].clear()
        return num_requests

    def num_passengers(self):
        n = 0
        for requests_set in self.requests.values():
            n += len(requests_set)
        return n

    def update_state(self):
        if self.state == ElevatorState.IDLE:
            if self.floor != self.target_floor:
                self.time_to_next_floor = TIME_PER_FLOOR
                if self.floor < self.target_floor:
                    self.state = ElevatorState.MOVING_UP
                else:
                    self.state = ElevatorState.MOVING_DOWN
        else:
            self.time_to_next_floor -= 1
            if self.time_to_next_floor == 0:
                # change self.floor
                if self.state == ElevatorState.MOVING_UP and self.floor < self.num_floors - 1:
                    self.floor += 1
                elif self.state == ElevatorState.MOVING_DOWN and self.floor > 0:
                    self.floor -= 1

                if self.floor == self.target_floor:
                    self.state = ElevatorState.IDLE
                else:
                    self.time_to_next_floor = TIME_PER_FLOOR

    def __repr__(self):
        return f"Elevator[{self.floor=}, {self.target_floor=}, {self.state=}, {self.time_to_next_floor=}, {self.requests=}]"
