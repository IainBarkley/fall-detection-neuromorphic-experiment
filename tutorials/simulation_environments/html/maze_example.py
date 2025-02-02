import nengo
import scipy.ndimage
import numpy as np
from functools import partial

from random_maze import RandomBlockMazeGenerator

model = nengo.Network(seed=13)

n_sensors = 20

def sense_to_ang_vel(x, n_sensors):
        
    rotation_weights = np.linspace(-1, 1, n_sensors)

    return np.dot(rotation_weights, np.array(x))

def generate_sensor_readings(map_arr,
                             n_sensors=30,
                             fov_rad=np.pi,
                             x=0,
                             y=0,
                             th=0,
                             max_sensor_dist=10,
                            ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    dists = np.zeros((n_sensors,))

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    for i, ang in enumerate(angs):
        dists[i] = get_collision_coord(map_arr, x, y, ang, max_sensor_dist)

    return dists


def get_collision_coord(map_array, x, y, th,
                        max_sensor_dist=10*4,
                        dr=0.05,
                       ):
    """
    Find the first occupied space given a start point and direction
    Meant for a zoomed in map_array
    """
    # Define the step sizes
    dx = np.cos(th) * dr
    dy = np.sin(th) * dr

    # Initialize to starting point
    cx = x
    cy = y

    for i in range(int(max_sensor_dist/dr)):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy
        #cx = np.clip(cx, 0, map_array.shape[0] - 1)
        #cy = np.clip(cy, 0, map_array.shape[1] - 1)
        if map_array[int(cx), int(cy)] == 1:
            return (i - 1)*dr

    return max_sensor_dist

class NengoMazeEnvironment(object):
    """
    Defines a maze environment to be used as a Nengo node
    Takes as input the agent x,y,th state as well as a map generation seed
    """

    def __init__(self,
                 n_sensors,
                 fov=180,
                 height=12,
                 width=12,
                 max_sensor_dist=10,
                 normalize_sensor_output=False,
                 input_type='directional_velocity',
                 dt=0.01,
                ):

        # Sets how inputs are interpreted
        # Forced to stay within the bounds
        assert(input_type) in ['position', 'holonomic_velocity', 'directional_velocity']
        self.input_type = input_type
        
        # dt value for the velocity inputs
        self.dt = dt

        # Number of distance sensors
        self.n_sensors = n_sensors

        self.max_sensor_dist = max_sensor_dist

        # If true, divide distances by max_sensor_dist
        self.normalize_sensor_output = normalize_sensor_output

        # Size of the map
        self.height = height
        self.width = width

        self.x = int(width/2.)
        self.y = int(height/2.)
        self.th = 0

        self.sensor_dists = np.zeros((self.n_sensors,))

        self.fov = fov
        self.fov_rad = fov * np.pi / 180.

        # Save the last seed used so as to not regenerate a new map until needed
        self.current_seed = 0

        # Create the default starting map
        self._generate_map()
        
        # Set up svg element templates to be filled in later
        self.tile_template = '<rect x={0} y={1} width=1 height=1 style="fill:black"/>'
        self.agent_template = '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:blue" transform="translate({0},{1}) rotate({2})"/>'
        self.sensor_template = '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:rgb(128,128,128);stroke-width:.1"/>'
        self.svg_header = '<svg width="100%%" height="100%%" viewbox="0 0 {0} {1}">'.format(self.height, self.width)

        self._generate_svg()

    def _generate_map(self):
        """
        Generate a new map based on the current seed
        """
        np.random.seed(self.current_seed)

        maze = RandomBlockMazeGenerator(maze_size=self.height - 2, # the -2 is because the outer wall gets added
                                        obstacle_ratio=.2,
                                       )
        self.map = maze.maze
    
    def _generate_svg(self):
        
        # draw tiles
        tiles = []
        for i in range(self.height):
            for j in range(self.width):
                # For simplicity and efficiency, only draw the walls and not the empty space
                # This will have to change when tiles can have different colours
                if self.map[i, j] == 1:
                    tiles.append(self.tile_template.format(i, j))

        # draw agent
        direction = self.th * 180. / np.pi + 90.
        x = self.x
        y = self.y
        th = self.th
        agent = self.agent_template.format(x, y, direction)

        svg = self.svg_header
        
        svg += ''.join(tiles)

        # draw distance sensors
        lines = []
        self.sensor_dists = generate_sensor_readings(
            map_arr=self.map,
            n_sensors=self.n_sensors,
            fov_rad=self.fov_rad,
            x=x,
            y=y,
            th=th,
            max_sensor_dist=self.max_sensor_dist,
        )
        ang_interval = self.fov_rad / self.n_sensors
        start_ang = -self.fov_rad/2. + th

        for i, dist in enumerate(self.sensor_dists):
            sx = dist*np.cos(start_ang + i*ang_interval) + self.x
            sy = dist*np.sin(start_ang + i*ang_interval) + self.y
            lines.append(self.sensor_template.format(self.x, self.y, sx, sy))
        svg += ''.join(lines)

        svg += agent
        svg += '</svg>'

        self._nengo_html_ = svg

    def free_space(self, x, y):
        nx = np.clip(int(np.floor(x)), 1, self.width - 1)
        ny = np.clip(int(np.floor(y)), 1, self.height - 1)
        return self.map[nx, ny] == 0

    def __call__(self, t, v):

        if self.input_type == 'holonomic_velocity':
            dx = v[0] * self.dt
            dy = v[1] * self.dt
        elif self.input_type == 'directional_velocity':
            #NOTE: the second input is unused in this case
            self.th += v[2] * self.dt
            dx = np.cos(self.th) * v[0] * self.dt
            dy = np.sin(self.th) * v[0] * self.dt
        elif self.input_type == 'position':
            dx = v[0] - self.x
            dy = v[1] - self.y
            self.th = v[2]

        # Only move if the new location is not a wall
        if self.free_space(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy
        
        # Keep the agent within the bounds
        self.x = np.clip(self.x, 1, self.width - 1)
        self.y = np.clip(self.y, 1, self.height - 1)
        self.th += v[2] * self.dt
        if self.th > 2*np.pi:
            self.th -= 2*np.pi
        elif self.th < -2*np.pi:
            self.th += 2*np.pi
        
        seed = int(v[3])

        if seed != self.current_seed:
            self.current_seed = seed
            self._generate_map()
        
        # Generate SVG image for nengo_gui
        # sensor_dists is also calculated in this function
        self._generate_svg()

        if self.normalize_sensor_output:
            self.sensor_dists /= self.max_sensor_dist

        #return self.sensor_dists
        return np.concatenate([[self.x], [self.y], [self.th], self.sensor_dists])


with model:

    map_selector = nengo.Node([0])
    
    linear_velocity = nengo.Node([0])

    angular_velocity = nengo.Ensemble(n_neurons=200, dimensions=1)

    environment = nengo.Node(
        NengoMazeEnvironment(
            n_sensors=n_sensors,
            height=12,
            width=12,
            fov=180,
        ),
        size_in=4,
        size_out=n_sensors + 3,
    )

    nengo.Connection(map_selector, environment[3])
    
    nengo.Connection(linear_velocity, environment[0], synapse=None)
    nengo.Connection(angular_velocity, environment[2], synapse=None)
    
    control_func = partial(sense_to_ang_vel, n_sensors=n_sensors)
    nengo.Connection(environment[3:], angular_velocity,
                     function=control_func)
