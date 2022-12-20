# ONLY FOR A SINGLE ELEVATOR SYSTEM

# For animation
from tkinter import *
import time

ELEVATOR_WIDTH_FACTOR = 3  # one third of building

LIGHT_BLUE = "#a2bff5"
LIGHT_RED = "#d94848"
LIGHT_GREEN = "#48d948"
GRAY = "#868786"

class ElevatorAnimation:
    def __init__(self, env, width: int = 600, height: int = 1000, padding: int = 70, delay: float = 0.5, title: str = ""):
        # environment properties
        self.env = env

        # drawing properties
        self.root = Tk()
        self.width = width      # canvas width
        self.height = height    # canvas height
        self.padding = padding  # padding between building and canvas edges
        self.thickness = 6      # thickness of outlines
        self.canvas = Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        # animation properties
        self.delay = delay   # delay between timesteps (in seconds)

        # other properties
        self.title = title
        self.timestep_text = None


    '''
    This function draws the title of the animation
    '''
    def draw_title(self):
        text_x = self.width // 2  # middle of canvas
        text_y = self.padding // 2  # middle of top padding
        self.canvas.create_text(text_x, text_y, text=self.title, fill="black", font=('Helvetica 18 bold'), justify=CENTER)


    '''
    This function draws the current timestep text
    '''
    def draw_timestep(self):
        text_x = self.width // 2  # middle of canvas
        text_y = self.height - self.padding // 2  # middle of bottom padding
        timestep = self.env.t
        timestep_text = f"Current Timestep: {timestep}"
        self.timestep_text = self.canvas.create_text(text_x, text_y, text=timestep_text, fill="black", font=('Helvetica 18 bold'), justify=CENTER)


    '''
    This function draws the building onto self.canvas
    
    Parameters:
    - building_width (int): The width of the building (in pixels)
    - building_height (int): The height of the building
    - num_floors (int): The number of floors in the building
    
    Returns:
    - tuple: A tuple containing (x1, y1, x2, y2), the coordinates of the building (top-left and bottom-right corners)
    '''
    def draw_building(self, building_width, building_height, num_floors):
        x_mid = self.width // 2    # middle of canvas
        y_mid = self.height // 2

        x1 = x_mid - (building_width // 2)  # top-left corner
        y1 = y_mid - (building_height // 2)

        x2 = x_mid + (building_width // 2)  # bottom-right corner
        y2 = y_mid + (building_height // 2)

        self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=GRAY, width=self.thickness)    # create building outline

        for i in range(num_floors):
            floor_height = building_height // num_floors
            y_add = i * floor_height
            floor_x1 = x1
            floor_y1 = y2 - y_add
            floor_x2 = x2
            floor_y2 = floor_y1
            self.canvas.create_line(floor_x1, floor_y1, floor_x2, floor_y2, fill="black", width=2)    # create line for each floor in building

            text = str(i)
            text_x = x1 - (self.padding // 3)
            text_y = floor_y1 - (floor_height // 2)
            self.canvas.create_text(text_x, text_y, text=text, fill="black", font=('Helvetica 18 bold'), justify=CENTER)

        # Draw vertical line indicating elevator shaft
        shaft_x1 = x1 + (building_width // ELEVATOR_WIDTH_FACTOR) + (self.thickness // 2)
        shaft_y1 = y1
        shaft_x2 = shaft_x1
        shaft_y2 = y2
        self.canvas.create_line(shaft_x1, shaft_y1, shaft_x2, shaft_y2, fill="black", width=2)

        return x1, y1, x2, y2   # return coordinates of building


    '''
    This function draws the elevator onto self.canvas
    
    Parameters:
    - building_coords (tuple): A tuple containing (x1, y1, x2, y2), the coordinates of the building
    
    Returns:
    - tuple: A tuple containing (elev_x1, elev_y1, elev_x2, elev_y2), the coordinates of the elevator
    '''
    def draw_elevator(self, building_coords):
        # Building coordinates
        x1, y1, x2, y2 = building_coords
        building_width = x2 - x1
        building_height = y2 - y1

        # Calculate where to draw the elevator
        elevator = self.env.elevators[0]
        cur_floor = elevator.floor
        num_floors = self.env.num_floors

        elevator_width = building_width // ELEVATOR_WIDTH_FACTOR     # 3 so that elevator is 1/3 width of the building
        elevator_height = building_height // num_floors

        elev_x1 = x1 + (self.thickness // 2)                          # top-left corner
        elev_y1 = (y2 - cur_floor * elevator_height) - elevator_height

        elev_x2 = elev_x1 + elevator_width      # bottom-right corner
        elev_y2 = elev_y1 + elevator_height

        self.canvas.create_rectangle(elev_x1, elev_y1, elev_x2, elev_y2, fill=LIGHT_BLUE, outline="black", width=2)

        return elev_x1, elev_y1, elev_x2, elev_y2       # return coordinates of elevator


    '''
    This function draws all requests (inside and outside of the elevator) onto self.canvas

    Parameters:
    - elev_coords (tuple): A tuple containing the coordinates of the elevator
    - building_coords (tuple): A tuple containing the coordinates of the building
    '''
    def draw_requests(self, elev_coords, building_coords):
        # ------------------------- DRAW REQUESTS (INSIDE ELEVATOR) ---------------------------
        elevator = self.env.elevators[0]
        requests = elevator.requests    # requests currently inside the elevator

        elev_x1, elev_y1, elev_x2, elev_y2 = elev_coords
        elevator_width = elev_x2 - elev_x1

        num_requests = 0
        for request_list in requests.values():
            num_requests += len(request_list)

        if num_requests > 0:
            request_width = min(elevator_width // num_requests, elevator_width // 4)   # limit to 1/4 width of elevator
        else:
            request_width = elevator_width // 4
        request_height = int(request_width * 2)   # make person slightly taller than wide

        i = 0
        for request_list in requests.values():
            for request in request_list:
                x_add = i * request_width
                request_x1 = elev_x1 + x_add
                request_y1 = elev_y2 - request_height
                request_x2 = request_x1 + request_width
                request_y2 = request_y1 + request_height
                self.canvas.create_oval(request_x1, request_y1, request_x2, request_y2, fill=LIGHT_GREEN, outline=LIGHT_GREEN)

                floor_text = str(request.target_floor)
                floor_text_x = request_x1 + (request_width // 2)
                floor_text_y = request_y1 + (request_height // 2)
                self.canvas.create_text(floor_text_x, floor_text_y, text=floor_text, fill="black", font=('Helvetica 15 bold'), justify=CENTER)

                i += 1

        # ------------------------- DRAW UNASSIGNED REQUESTS (OUTSIDE ELEVATOR) ---------------------------
        unassigned_requests = self.env.unassigned_requests
        building_x1, building_y1, building_x2, building_y2 = building_coords
        num_floors = self.env.num_floors

        building_height = building_y2 - building_y1
        floor_height = building_height // num_floors

        for f in range(num_floors):
            u_request_list = unassigned_requests.get(f, [])

            num_u_requests = len(u_request_list)
            if num_u_requests > 0:
                # distance between right wall of elevator and right wall of building, divided by the number of u-requests on floor f
                u_request_width = min((building_x2 - elev_x2) // num_u_requests, (building_x2 - elev_x2) // 8)
            else:
                u_request_width = (building_x2 - elev_x2) // 8
            u_request_height = int(u_request_width * 2)     # person is twice as tall as they are wide

            u_request_y1 = building_y2 - (f * floor_height)
            u_request_y2 = u_request_y1 - u_request_height

            i = 0
            for u_request in u_request_list:
                x_add = i * u_request_width
                u_request_x1 = elev_x2 + x_add
                u_request_x2 = u_request_x1 + request_width
                self.canvas.create_oval(u_request_x1, u_request_y1, u_request_x2, u_request_y2, fill=LIGHT_RED, outline=LIGHT_RED)

                floor_text = str(u_request.target_floor)
                floor_text_x = u_request_x1 + (u_request_width // 2)
                floor_text_y = u_request_y1 - (u_request_height // 2)
                self.canvas.create_text(floor_text_x, floor_text_y, text=floor_text, fill="black", font=('Helvetica 15 bold'), justify=CENTER)

                i += 1

    '''
    This function draws the entire environment (building, elevator, requests, title/timesteps)
    '''
    def draw_environment(self):
        self.canvas.delete("all")   # clear existing drawing before drawing updated environment
        # if self.timestep_text is not None:
        #     self.canvas.delete(self.timestep_text)

        # Draw title
        self.draw_title()

        # Draw timestep text
        self.draw_timestep()

        # Draw building
        building_width = self.width - (self.padding * 2)
        building_height = self.height - (self.padding * 2)
        num_floors = self.env.num_floors
        building_coords = self.draw_building(building_width, building_height, num_floors)

        # Draw elevator
        elev_coords = self.draw_elevator(building_coords)

        # Draw requests (all people waiting inside and outside elevator)
        self.draw_requests(elev_coords, building_coords)

        # display changes
        self.update()

    def update(self):
        self.root.update()
        time.sleep(self.delay)

    def set_environment(self, env):
        self.env = env
