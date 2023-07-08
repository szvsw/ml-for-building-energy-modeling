import logging

from os import PathLike
from typing import List, Union

import taichi as ti
from taichi.algorithms._algorithms import PrefixSumExecutor
import numpy as np
import geopandas as gpd
import pandas as pd

from shapely import Polygon, LineString, MultiPoint, Point

# ti.init(arch=ti.gpu, device_memory_fraction=0.7, kernel_profiler=True, debug=True)
# ti.init(arch=ti.cpu, kernel_profiler=True)
ti.init(arch=ti.gpu, device_memory_fraction=0.5, kernel_profiler=True)
uint1 = ti.types.quant.int(1, signed=False)

logging.basicConfig()
logger = logging.getLogger("Radiation Analysis")
logger.setLevel(logging.INFO)

# TODO: deal with collinear edges which result in sensors inside neighboring building!


# TODO: migrate node_heights field to a Node.field()
@ti.dataclass
class Node:
    height: float


@ti.dataclass
class Edge:
    # building_id: ti.int16 # TODO: add an assertion which checks for fewer than 2**16-1 buildings

    start: ti.math.vec2
    end: ti.math.vec2
    slopevec: ti.math.vec2
    slope: float
    normal: ti.math.vec2
    normal_theta: float
    az_start_angle: float
    height: float  # TODO: This could be rounded to save memory, e.g. uint16, or a quantized datatype e.g. uint10
    n_floors: ti.int8

    sensor_start_ix: int  # TODO: should these be forced to 64 bit?
    sensor_end_ix: int
    sensor_ct: int


@ti.dataclass
class XYSensor:
    hit_count: int
    loc: ti.math.vec2
    parent_edge_id: int

    xyz_sensor_start_ix: int  # TODO: should these be forced to 64 bit?
    xyz_sensor_ct: int


@ti.dataclass
class XYZSensor:
    height: float
    rad: float
    parent_sensor_id: int


@ti.dataclass
class Hit:
    loc_x_ix: ti.i16
    loc_y_ix: ti.i16
    height: float
    distance: float

    @ti.func
    def centroid(self) -> ti.math.vec2:
        return ti.Vector(
            [self.loc_x_ix + 0.5, self.loc_y_ix + 0.5]
        )  # TODO: assumes a bin spacing of 1m!


N_LOOPS_TO_UNROLL = 1


@ti.data_oriented
class Tracer:
    node_width: float  # meters
    sensor_inset: float  # meters
    sensor_spacing: float  # meters
    f2f_height: float  # floor-to-floor height, meters
    max_ray_length: float  # meters
    ray_step_size: float  # meters
    n_ray_steps: int

    depth: int  # quadtree level count
    levels: List[ti.SNode]  # pointers to each 2x2 level of the quadtree
    node_heights: ti.ScalarField  # stores the height of each populated node in the quadtree

    # TODO: make sure all class attrs are represented here
    edge_starts: ti.ScalarField
    edge_ends: ti.ScalarField
    edge_slopes: ti.ScalarField
    edge_heights: ti.ScalarField
    edge_normals: ti.ScalarField

    gdf: gpd.GeoDataFrame
    height_col: str
    id_col: str

    def __init__(
        self,
        filepath: Union[str, bytes, PathLike],
        height_col: str,
        id_col: str,
        node_width: float = 1,
        sensor_inset: float = 0.5,
        sensor_spacing: float = 1,
        f2f_height: float = 3,
        max_ray_length: float = 400.0,
        ray_step_size: float = 1.0,
        convert_crs=False,
    ):
        # TODO: add meter conversion, better crs validation/conversion
        # store the bin size in meters
        global N_LOOPS_TO_UNROLL
        assert (
            node_width == 1
        ), "Currently only supports dividing the space into 1 m node widths"
        self.node_width = node_width
        padding = 5 * node_width

        # store the sensor grid config
        self.sensor_inset = sensor_inset
        self.sensor_spacing = sensor_spacing
        self.f2f_height = f2f_height
        self.max_ray_length = max_ray_length
        self.ray_step_size = ray_step_size
        self.n_ray_steps = int(self.max_ray_length / ray_step_size)
        self.steps_per_unroll_loop = 100  # TODO: add to init args
        N_LOOPS_TO_UNROLL = int(
            np.ceil(self.n_ray_steps / self.steps_per_unroll_loop)
        )  # TODO: this should be a class property but it throws an error in the static unroll command in the kernel if so

        # Load the GDF
        self.gdf: gpd.GeoDataFrame = gpd.read_file(
            filepath
        )  # nb: assumes a flattened projection already holds.
        if convert_crs:
            logger.info("Converting crs...")
            self.gdf = self.gdf.to_crs("EPSG:32633")
        self.height_col = height_col
        self.id_col = id_col
        assert (
            height_col in self.gdf.columns
        ), f"The supplied height column '{height_col}' was not found in the GDF's columns: {self.gdf.columns}"
        assert (
            id_col in self.gdf.columns
        ), f"The supplied unique ID column '{id_col}' was not found in the GDF's columns: {self.gdf.columns}"

        # compute number of floors
        self.gdf["N_FLOORS"] = np.ceil(self.gdf[height_col].values / self.f2f_height)

        # find the bbox
        x_low, y_low, x_high, y_high = self.gdf.geometry.total_bounds

        # relocate the gdf, with a little bit of wiggle room around the origin axes
        self.gdf.geometry = self.gdf.geometry.translate(
            -x_low + padding, -y_low + padding
        )

        base_gdf = self.gdf.copy()
        tile_ct = 0
        for i in range(tile_ct):
            for j in range(tile_ct):
                new_gdf = base_gdf.copy()
                new_gdf.geometry = new_gdf.geometry.translate(
                    ((x_high - x_low) + 2 * padding) * (i + 1),
                    ((y_high - y_low) + 2 * padding) * (j + 1),
                )
                self.gdf = pd.concat([self.gdf, new_gdf], axis=0, ignore_index=True)

        # compute the approx. number of sensors, ignoring sensor inset
        approx_n_sensors = int(
            np.floor(
                (
                    self.gdf.geometry.boundary.length
                    * self.gdf["N_FLOORS"]
                    / self.sensor_spacing
                ).sum()
            )
        )
        logger.info(f"Building Count: {len(self.gdf)}")
        logger.info(f"Approximate Sensor Count: {approx_n_sensors}")

        # find the new bbox
        x_low, y_low, x_high, y_high = self.gdf.geometry.total_bounds

        # check if length or width controls
        length = y_high + padding
        width = x_high + padding
        max_dim = length if length > width else width
        self.length = length
        self.width = width

        # compute how many bins are required in the longer axis
        min_nodes_required = int(np.ceil(max_dim / self.node_width))

        # compute how many binary divisions are needed in the longer axis
        self.depth = int(
            np.ceil(np.log2(min_nodes_required))
        )  # TODO: possibly not necessary if not using a quadtree
        assert (
            self.depth < 16
        ), f"Currently only supports quadtrees with a depth < 16, and a depth of {self.depth} is required for the {length,width} bbox"
        logger.info(f"QuadTree Depth: {self.depth}")

        # Create Scene Tree
        # TODO: explore performance and memory implications of using bitmasked instead
        self.levels = [ti.root.bitmasked(ti.ij, (2**self.depth, 2**self.depth))]
        # self.levels = []
        # root = ti.root.pointer(ti.ij, (2, 2))
        # self.levels.append(root)
        # for i in range(self.depth - 1):
        #     level = self.levels[-1].pointer(ti.ij, (2, 2))
        #     self.levels.append(level)

        # create a struct field and place it
        self.node_heights = ti.field(float)
        self.tree_leaves.place(self.node_heights)

        # Extract edges into Fields
        logger.info("Extracting edges...")
        self.extract_flat_edge_list()

        # add edges to quadtree
        logger.info("Populating tree...")
        self.add_edges_to_tree()
        ti.sync()

        sensor_count = self.edge_sensor_parent_ix.shape[0]
        logger.info(f"XY sensor count: {sensor_count}")

        # Build a dynamic list of hits per ray
        logger.info("Building dynamic hit tracking data structure...")
        self.n_azimuths = 48  # TODO: make this an init arg
        self.azimuth_inc = 2 * np.pi / (self.n_azimuths * 2)
        self.sensor_root = ti.root.dense(ti.i, sensor_count)
        self.ray_root = self.sensor_root.dense(ti.j, self.n_azimuths)
        self.hit_block = self.ray_root.dynamic(
            ti.k,
            2 ** (int(np.ceil(np.log2(self.n_ray_steps)))),
            chunk_size=64,
        )  # TODO: big performance hit on gpu
        self.hits = Hit.field()
        self.hit_block.place(self.hits)

        # Build
        self.n_elevations = 16  # TODO: make this an init arg
        self.elevation_inc = 0.5 * np.pi / self.n_elevations

        # Init xy sensor locations
        logger.info("Initializing xy-plane sensors...")
        self.xy_sensors = (
            XYSensor.field()
        )  # TODO: Why do I have to place hit_block first for compilation to work?
        self.sensor_root.place(self.xy_sensors)
        self.init_xy_sensors()
        ti.sync()

        # Determine how many xyz sensors are needed for data collection
        xyz_cts = self.xy_sensors.xyz_sensor_ct.to_numpy()
        xyz_ends = np.cumsum(xyz_cts)  # use cumulative sums so that a sensor
        xyz_starts = np.roll(xyz_ends, shift=1)
        xyz_starts[0] = 0
        self.xy_sensors.xyz_sensor_start_ix.from_numpy(xyz_starts)
        xy_sensor_parent_ix = np.repeat(
            np.arange(xyz_cts.shape[0]), xyz_cts.astype(int)
        )

        xyz_sensor_count = xy_sensor_parent_ix.shape[0]
        logger.info(f"XYZ sensor count: {xyz_sensor_count}")
        logger.info("Initializing xyz sensors...")
        self.xyz_sensors = XYZSensor.field()
        self.xyz_sensor_root = ti.root.dense(ti.i, xyz_sensor_count)
        self.xyz_sensor_root.place(self.xyz_sensors)
        self.xyz_view_root = self.xyz_sensor_root.bitmasked(
            ti.jk, (self.n_azimuths, self.n_elevations)
        )

        self.xyz_views = ti.field(
            dtype=uint1
        )  # TODO: use a quantized data type with a single bit (uint1)
        self.ui1bitpacker = ti.BitpackedFields(max_num_bits=32)
        self.ui1bitpacker.place(self.xyz_views)
        self.xyz_view_root.place(self.ui1bitpacker)
        self.xyz_sensors.parent_sensor_id.from_numpy(xy_sensor_parent_ix)

        self.init_xyz_sensors()
        ti.sync()
        # TODO: add n_azimuths and then n_elevations tree branches to skip compute

        logger.info(f"XY rays: {sensor_count * self.n_azimuths}")
        xyz_ray_ct = xyz_sensor_count * self.n_azimuths * self.n_elevations
        assert (
            xyz_ray_ct < 2**32
        ), f"This scene requires {xyz_ray_ct} rays which is greater than the currently supported max of 2^32 ~= 4e9."
        logger.info(f"XYZ rays: {xyz_ray_ct}")

        # # Ray trace in xy plane
        # logger.info("XY tracing...")
        # # self.xy_trace_divergent()
        # self.xy_trace()
        # ti.sync()
        # logger.info("XY tracing complete.")

        # # Ray trace using xyz data
        # logger.info("XYZ tracing...")
        # self.xyz_trace()
        # ti.sync()
        # logger.info("XYZ tracing complete.")

        # Ray trace using xyz data
        logger.info("XYZ tracing...")
        self.xyz_trace_unified()
        ti.sync()
        logger.info("XYZ tracing complete.")

        self.sensor_3d_points = ti.Vector.field(3, dtype=float, shape=xyz_sensor_count)
        self.sensor_3d_colors = ti.Vector.field(3, dtype=float, shape=xyz_sensor_count)
        self.sensor_3d_rays = ti.Vector.field(
            3, dtype=float, shape=2 * (self.n_azimuths * self.n_elevations)
        )
        self.load_3d_points()
        self.load_3d_sensor_rays(0)

        ti.profiler.print_kernel_profiler_info()

    def init_gui(self):
        self.window = ti.ui.Window("UMI RayTrace", (1024, 1024), pos=(100, 100))
        self.gui = self.window.get_gui()
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.up(0, 1, 0)
        self.camera.position(0, 10, 0)
        self.camera.lookat(1, 10, 1)

    def render_scene(self):
        sensor_ix = 0
        controls_changed = True
        while self.window.running:
            with self.gui.sub_window("Sensor selector", 0.1, 0.1, 0.8, 0.15):
                old_ix = sensor_ix
                sensor_ix = self.gui.slider_int(
                    text="Sensor Index",
                    old_value=sensor_ix,
                    minimum=0,
                    maximum=self.xyz_sensors.shape[0],
                )
                if old_ix != sensor_ix:
                    controls_changed = True

                if controls_changed:
                    self.load_3d_sensor_rays(sensor_ix)
                    controls_changed = False
            self.camera.track_user_inputs(self.window, hold_key=ti.ui.RMB)
            self.scene.ambient_light((1, 1, 1))
            self.scene.particles(
                self.sensor_3d_points,
                radius=0.2,
                per_vertex_color=self.sensor_3d_colors,
            )
            self.scene.lines(self.sensor_3d_rays, width=1, color=(1, 1, 1))
            self.scene.set_camera(self.camera)
            self.canvas.scene(self.scene)
            self.window.show()

    def extract_flat_edge_list(self):
        """
        Extracts all edges to a flattened list
        # TODO: figure out if there is a vectorized manner instead of having to do it bldg by bldg
        """
        starts = []
        ends = []
        run_rises = []
        heights = []
        n_floors = []
        normals = []
        building_ids = []
        normal_fails = 0
        for i, _geom in enumerate(self.gdf.geometry):
            # Manual explode of geometry for multiploygon handling
            for geom in _geom.geoms if type(_geom) != Polygon else [_geom]:
                # Get the points from the boundary
                # shapely poly linestrings are closed, so we don't need the repeated point
                points = np.array(geom.boundary.coords)[:-1]

                # Roll the points over
                next_points = np.roll(points, shift=-1, axis=0)

                # compute the slope components and unitize
                run_rise = next_points - points
                run_rise = run_rise / np.linalg.norm(run_rise, axis=1).reshape(-1, 1)

                # compute the normals for each edge
                # if we imagine the line segment as part of a plane which is perp
                # to the xy plane then we can take the cross product of the slope
                # and the k unit vector to get the perp vector which points to the outside.
                # note that this should never point inside because of the right-hand rule.
                cross_test = np.zeros(shape=(run_rise.shape[0], 3))
                cross_test[:, :2] = run_rise.copy()
                up = np.zeros_like(cross_test)
                up[:, 2] = 1
                normal = np.cross(
                    up, cross_test, axisa=1, axisb=1
                )  # this should point outward due to the winding of the polygons.
                normal = normal[:, :2]

                # some debug code for checking normal orientation
                # test_point = (points + next_points) / 2 + normal * 0.1 # move it 1 cm inside
                # for j in range(test_point.shape[0]):
                #     try:
                #         assert not geom.contains(Point(test_point[j,0], test_point[j,1]))
                #     except AssertionError:
                #         normal_fails = normal_fails + 1

                # append computed properties
                starts.append(points)
                ends.append(next_points)
                run_rises.append(run_rise)
                heights.append(
                    np.ones(points.shape[0]) * self.gdf[self.height_col][i]
                )  # TODO: these could be found via a building parent ref
                n_floors.append(np.ones(points.shape[0]) * self.gdf["N_FLOORS"][i])
                normals.append(normal)

        # Create flattened list of all edge data (i.e. flattened over buildings axis)
        starts = np.vstack(starts)
        ends = np.vstack(ends)
        run_rises = np.vstack(run_rises)
        normals = np.vstack(normals)
        heights = np.concatenate(heights)
        n_floors = np.concatenate(n_floors)

        # Determine necessary sensor count per edge
        lengths = np.linalg.norm(starts - ends, axis=1)
        sensor_counts = (lengths - 2 * self.sensor_inset) / self.sensor_spacing
        sensor_counts = np.floor(np.where(sensor_counts >= 1, sensor_counts + 1, 0))
        sensor_ends = np.cumsum(sensor_counts)
        sensor_starts = np.roll(sensor_ends, shift=1)
        sensor_starts[0] = 0

        # Create the fields
        self.edge_starts = ti.field(float, shape=starts.shape)
        self.edge_ends = ti.field(float, shape=ends.shape)
        self.edge_slopes = ti.field(float, shape=run_rises.shape)
        self.edge_heights = ti.field(float, shape=heights.shape)
        self.edge_normals = ti.field(float, shape=normals.shape)
        self.edge_n_floors = ti.field(float, shape=n_floors.shape)
        self.edge_sensor_starts = ti.field(int, shape=sensor_starts.shape)
        self.edge_sensor_ends = ti.field(int, shape=sensor_ends.shape)
        self.edge_sensor_counts = ti.field(int, shape=sensor_counts.shape)

        # Copy the numpy data over
        self.edge_starts.from_numpy(starts)
        self.edge_ends.from_numpy(ends)
        self.edge_slopes.from_numpy(run_rises)
        self.edge_heights.from_numpy(heights)
        self.edge_n_floors.from_numpy(n_floors)
        self.edge_normals.from_numpy(normals)
        self.edge_sensor_starts.from_numpy(sensor_starts)
        self.edge_sensor_ends.from_numpy(sensor_ends)
        self.edge_sensor_counts.from_numpy(sensor_counts)

        # Create the semantic edge objects in a dense struct field for better memory access
        self.edges = Edge.field(shape=self.edge_heights.shape)
        self.make_edges()
        ti.sync()

        # This array will allow non-divergent sensor construction
        # It has one row per sensor which lets the sensor identify the parent edge
        sensor_parent_ix = np.repeat(
            np.arange(sensor_counts.shape[0]), sensor_counts.astype(int)
        )
        self.edge_sensor_parent_ix = ti.field(int, shape=sensor_parent_ix.shape)
        self.edge_sensor_parent_ix.from_numpy(sensor_parent_ix)

    @ti.kernel
    def make_edges(self):
        for edge_ix in self.edges:
            # extract the endpoints/data
            x0 = self.edge_starts[edge_ix, 0]
            y0 = self.edge_starts[edge_ix, 1]
            x1 = self.edge_ends[edge_ix, 0]
            y1 = self.edge_ends[edge_ix, 1]
            xn = self.edge_normals[edge_ix, 0]
            yn = self.edge_normals[edge_ix, 1]
            xm = self.edge_slopes[edge_ix, 0]
            ym = self.edge_slopes[edge_ix, 1]
            h = self.edge_heights[edge_ix]
            n_floors = self.edge_n_floors[edge_ix]

            sensor_start_ix = self.edge_sensor_starts[edge_ix]
            sensor_end_ix = self.edge_sensor_ends[edge_ix]
            sensor_ct = self.edge_sensor_counts[edge_ix]

            # make vectors
            normal = ti.Vector([xn, yn])
            start = ti.Vector([x0, y0])
            end = ti.Vector([x1, y1])
            slopevec = ti.Vector([xm, ym])

            # compute slope
            slope = ym / xm  # TODO: handle vert/hor lines

            # Compute the normal angle
            normal_theta = ti.atan2(yn, xn)

            # Compute the azimuth start angle for any sensor placed on this edge
            az_start_angle = normal_theta - np.pi * 0.5

            # Create the edge object
            self.edges[edge_ix] = Edge(
                start=start,
                end=end,
                slopevec=slopevec,
                slope=slope,
                normal=normal,
                normal_theta=normal_theta,
                az_start_angle=az_start_angle,
                height=h,
                n_floors=n_floors,
                sensor_start_ix=sensor_start_ix,
                sensor_end_ix=sensor_end_ix,
                sensor_ct=sensor_ct,
            )

    @ti.kernel
    def add_edges_to_tree(self):
        """
        This function determines where each line crosses a node threshold and
        updates that node's height accordingly.
        """
        for edge_ix in self.edges:
            # TODO: Update if edges switch to a dataclass representation
            # extract the endpoints/data
            edge = self.edges[edge_ix]
            x0 = edge.start.x
            y0 = edge.start.y
            x1 = edge.end.x
            y1 = edge.end.y
            h = edge.height
            slope = edge.slope

            # Sort the end points
            x_min = ti.min(x0, x1)
            x_max = ti.max(x0, x1)
            y_min = ti.min(y0, y1)
            y_max = ti.max(y0, y1)

            # Find the nearest node boundary line
            # TODO: currently only supports node width of 1!
            x_start = ti.ceil(x_min, dtype=int)
            x_end = ti.floor(x_max, dtype=int)
            y_start = ti.ceil(y_min, dtype=int)
            y_end = ti.floor(y_max, dtype=int)

            # Count the number of grid lines to check
            n_x_thresholds = 1 + x_end - x_start
            n_y_thresholds = 1 + y_end - y_start

            # Compute the grid line crossings and update node heights
            # TODO: DIVERGENCE WARNING
            # TODO: DIVERGENCE WARNING
            # TODO: DIVERGENCE WARNING
            # TODO: DIVERGENCE WARNING
            # but it's not so bad :)
            for x_int_ix in range(n_x_thresholds):
                x = x_start + x_int_ix  # TODO: currently only supports node width of 1
                y = slope * (x - x0) + y0
                y_ix = ti.floor(y, int)  # TODO: currently only supports node width of 1

                # Add height to quadtree if the edge is taller than the existing edge
                ti.atomic_max(self.node_heights[x - 1, y_ix], h)  # update left node
                ti.atomic_max(self.node_heights[x, y_ix], h)  # update right node

            for y_int_ix in range(n_y_thresholds):
                y = y_start + y_int_ix  # TODO: currently only supports node width of 1
                x = (1 / slope) * (y - y0) + x0
                x_ix = ti.floor(x, int)  # TODO: currently only supports node width of 1

                # Add height to quadtree if the edge is taller than the existing edge
                ti.atomic_max(self.node_heights[x_ix, y - 1], h)  # update lower node
                ti.atomic_max(self.node_heights[x_ix, y], h)  # update upper node

    @ti.kernel
    def init_xy_sensors(self):
        """
        Configures the location/parent information for every sensor by copying data over
        Relies on the repeat edge field edge_sensor_parent_ix
        """
        for sensor_ix in self.xy_sensors:
            # Locate the sensor in the original field and copy the parent id over
            parent_id = self.edge_sensor_parent_ix[sensor_ix]
            edge = self.edges[parent_id]

            # get the parent slope over
            slope = edge.slopevec

            # Determine the inset edge gap for the sensor
            start_gap = (
                slope * self.sensor_inset
            )  # TODO: this could be stored with parent # TODO: this could be centered

            # Determine which sensor this is along a the parent edge
            gap_ct = sensor_ix - edge.sensor_start_ix

            # compute the distance from the edge start vertex
            distance = start_gap + gap_ct * slope * self.sensor_spacing

            # Copy the parent edge start vertex over
            start_loc = edge.start

            # Copy the parent edge normal over
            normal = edge.normal

            # Set the new location by moving along edge the appropriate amount
            # and then 1.5m away from the wall following the normal
            self.xy_sensors[sensor_ix].loc = (
                start_loc + distance + normal * 1.5
            )  # TODO: make the offset distance a class attr

            # Store the parent id
            self.xy_sensors[sensor_ix].xyz_sensor_ct = edge.n_floors

            # Store the parent id
            self.xy_sensors[sensor_ix].parent_edge_id = parent_id

    @ti.kernel
    def init_xyz_sensors(self):
        for sensor_ix in self.xyz_sensors:
            parent_id = self.xyz_sensors[sensor_ix].parent_sensor_id
            xy_sensor = self.xy_sensors[parent_id]
            floor_ix = sensor_ix - xy_sensor.xyz_sensor_start_ix
            # Sensors should be in floor midpoint, so use 1.5xf2f
            height = floor_ix * 1.5 * self.f2f_height
            self.xyz_sensors[sensor_ix].height = height

    @ti.kernel
    def xy_trace(self):
        ray_step_size = self.ray_step_size  # TODO: will cause duplicate collisions
        steps_per_loop = self.steps_per_unroll_loop

        # Break ray stepping up into portions of ray
        # In order to prevent ti.ndrange overflow (there may be several billion checks to make even in 2d)
        # Use loop unnrolling via ti.static to keep inner loop parallelized
        for loop_ix in ti.static(range(N_LOOPS_TO_UNROLL)):
            step_offset = loop_ix * steps_per_loop
            for sensor_ix, az_ix, ray_step_ix in ti.ndrange(
                self.xy_sensors.shape[0], self.n_azimuths, steps_per_loop
            ):
                # Compute the rays's azimuth angle
                sensor = self.xy_sensors[sensor_ix]

                az_angle = (
                    self.azimuth_inc * az_ix
                    + self.edges[sensor.parent_edge_id].az_start_angle
                )

                # Compute the ray's xy-plane slope
                dx = ti.cos(
                    az_angle
                )  # TODO: precompute as a lookup in init based off of n_azimuths?
                dy = ti.sin(az_angle)
                slope = ti.Vector([dx, dy])

                # Get the ray's starting point
                start = sensor.loc

                # Length of ray to check
                distance = ray_step_size * (ray_step_ix + step_offset)

                # Initializing the next location to check
                next_loc = start + distance * slope

                # Tester for ray termination
                in_domain = (
                    (next_loc.x > 0)
                    and (next_loc.y > 0)
                    and (next_loc.x < self.width)
                    and (next_loc.y < self.length)
                )
                if in_domain:
                    # Get ray terminus node index
                    x_loc_ix = ti.floor(
                        next_loc.x, int
                    )  # TODO: assumes grid spacing = 1
                    y_loc_ix = ti.floor(next_loc.y, int)

                    # Check if node is active
                    if ti.is_active(self.tree_leaves, [x_loc_ix, y_loc_ix]) == 1:
                        # Get the node height and register a hit
                        node_height = self.node_heights[x_loc_ix, y_loc_ix]
                        # TODO: this is causing a large performance hit on gpu backend
                        self.hits[sensor_ix, az_ix].append(
                            Hit(
                                loc_x_ix=x_loc_ix,
                                loc_y_ix=y_loc_ix,
                                height=node_height,
                                distance=distance,  # TODO: should this use the node centroid distance instead?
                            )  # TODO: assumes a  grid spacing = 1
                        )
                        self.xy_sensors[sensor_ix].hit_count += 1

    @ti.kernel
    def xy_trace_divergent(self):
        max_ray_length = self.max_ray_length
        ray_step_size = self.ray_step_size  # TODO: will cause duplicate collisions

        for sensor_ix, az_ix in ti.ndrange(self.xy_sensors.shape[0], self.n_azimuths):
            # Compute the rays's azimuth angle
            sensor = self.xy_sensors[sensor_ix]

            az_angle = (
                self.azimuth_inc * az_ix
                + self.edges[sensor.parent_edge_id].az_start_angle
            )

            # Compute the ray's xy-plane slope
            dx = ti.cos(
                az_angle
            )  # TODO: precompute as a lookup in init based off of n_azimuths?
            dy = ti.sin(az_angle)
            slope = ti.Vector([dx, dy])

            # Get the ray's starting point
            start = sensor.loc

            # Tracker for ray extension
            ray_step_ix = 0.0

            # Initializing the next location to check
            distance = ray_step_ix * ray_step_size
            next_loc = start + distance * slope

            # Tester for ray termination
            in_domain = (
                (next_loc.x > 0)
                and (next_loc.y > 0)
                and (next_loc.x < self.width)
                and (next_loc.y < self.length)
                and distance < max_ray_length
            )
            while in_domain:
                # Get ray terminus node index
                x_loc_ix = ti.floor(next_loc.x, int)  # TODO: assumes grid spacing = 1
                y_loc_ix = ti.floor(next_loc.y, int)

                # Check if node is active
                if ti.is_active(self.tree_leaves, [x_loc_ix, y_loc_ix]) == 1:
                    # Get the node height and register a hit
                    node_height = self.node_heights[x_loc_ix, y_loc_ix]
                    # TODO: this is causing a large performance hit on gpu backend
                    self.hits[sensor_ix, az_ix].append(
                        Hit(
                            loc_x_ix=x_loc_ix,
                            loc_y_ix=y_loc_ix,
                            height=node_height,
                            distance=distance,  # TODO: should this use the node centroid distance instead?
                        )  # TODO: assumes a  grid spacing = 1
                    )
                    self.xy_sensors[sensor_ix].hit_count += 1

                # Advance the ray stepper
                ray_step_ix = ray_step_ix + 1.0

                # Compute the new length
                distance = ray_step_ix * ray_step_size

                # Compute the new location
                next_loc = start + distance * slope

                # Tester for ray termination
                in_domain = (
                    (next_loc.x > 0)
                    and (next_loc.y > 0)
                    and (next_loc.x < self.width)
                    and (next_loc.y < self.length)
                    and distance < max_ray_length
                )

    @ti.kernel
    def xyz_trace(self):
        for sensor_ix, az_ix, el_ix in ti.ndrange(
            self.xyz_sensors.shape[0], self.n_azimuths, self.n_elevations
        ):
            # get the xyz sensors corresponding xy sensor
            parent_sensor_id = self.xyz_sensors[sensor_ix].parent_sensor_id

            # get the xyz sensor's height
            xyz_sensor_height = self.xyz_sensors[sensor_ix].height

            # determine how many hits need to be checked based off of xy sensors hit table
            # n_hits_to_check = self.hits[parent_sensor_id, az_ix].length()
            n_hits_to_check = self.xy_sensors[parent_sensor_id].hit_count
            el_angle = (
                el_ix * self.elevation_inc
            )  # TODO: precompute these? or store slopes?

            # Initiate an iterator so we can bail out early
            # via a while loop, rather than using automatic iteration
            hit_ix = 0
            # create a flag for when a hit has been found
            hit_found = 0
            while hit_ix < n_hits_to_check and hit_found != 1:
                # Extract an xy hit and its properties
                hit = self.hits[parent_sensor_id, az_ix, hit_ix]
                hit_height = hit.height
                hit_distance = hit.distance
                # compute the height diff for the current xyz sensor
                height_diff = hit_height - xyz_sensor_height

                # compute the angle
                theta = ti.atan2(
                    height_diff, hit_distance
                )  # TODO: would using a slope divison be more performant?

                # Check if the sensor-to-other-building angle is greater than the sensor-to-sky-patch angle
                if theta > el_angle:
                    # Indicate a bail out if the building is obstructing
                    hit_found = 1

                # increment the hit ix iterator
                hit_ix = hit_ix + 1

            # If no obstructions found, then add the result in
            if hit_found != 1:
                self.xyz_sensors[sensor_ix].rad += 1  # TODO: look up sky matrix
                # Store a hit mask
                self.xyz_views[sensor_ix, az_ix, el_ix] = 1

    @ti.kernel
    def xyz_trace_unified(self):
        for sensor_ix, az_ix, el_ix in ti.ndrange(
            self.xyz_sensors.shape[0], self.n_azimuths, self.n_elevations
        ):
            # get the xyz sensors corresponding xy sensor
            parent_sensor_id = self.xyz_sensors[sensor_ix].parent_sensor_id
            parent_sensor = self.xy_sensors[parent_sensor_id]

            # get the xyz sensor's height
            xyz_sensor_height = self.xyz_sensors[sensor_ix].height

            el_angle = (
                el_ix * self.elevation_inc
            )  # TODO: precompute these? or store slopes?

            az_angle = (
                self.azimuth_inc * az_ix
                + self.edges[parent_sensor.parent_edge_id].az_start_angle
            )

            # Compute the ray's xy-plane slope
            dx = ti.cos(
                az_angle
            )  # TODO: precompute as a lookup in init based off of n_azimuths?
            dy = ti.sin(az_angle)
            slope = ti.Vector([dx, dy])

            # Get the ray's starting point
            start = parent_sensor.loc

            distance = self.trace_xyz_ray(start, slope, el_angle, xyz_sensor_height)

            # If no obstructions found, then add the result in
            if distance < 0:
                self.xyz_sensors[sensor_ix].rad += 1  # TODO: look up sky matrix
                # Store a hit mask
                self.xyz_views[sensor_ix, az_ix, el_ix] = 1
                # TODO: track hit location

    @ti.func
    def trace_xyz_ray(
        self,
        start: ti.math.vec2,
        slope: ti.math.vec2,
        el_angle: float,
        xyz_sensor_height: float,
    ) -> float:
        # Tracker for ray extension
        ray_step_ix = 0.0

        # Initializing the next location to check
        distance = ray_step_ix * self.ray_step_size
        next_loc = start + distance * slope

        # Tester for ray termination
        in_domain = (
            (next_loc.x > 0)
            and (next_loc.y > 0)
            and (next_loc.x < self.width)
            and (next_loc.y < self.length)
            and distance < self.max_ray_length
        )

        hit_found = 0
        while in_domain and hit_found != 1:
            # Get ray terminus node index
            x_loc_ix = ti.floor(next_loc.x, int)  # TODO: assumes grid spacing = 1
            y_loc_ix = ti.floor(next_loc.y, int)

            # Check if node is active
            if ti.is_active(self.tree_leaves, [x_loc_ix, y_loc_ix]) == 1:
                # Get the height of the node in the xy plane
                node_height = self.node_heights[x_loc_ix, y_loc_ix]

                # Compute the height difference to the edge crossed
                height_diff = node_height - xyz_sensor_height

                # compute the angle
                theta = ti.atan2(
                    height_diff, distance
                )  # TODO: would using a slope divison be more performant?

                # Check if the sensor-to-other-building angle is greater than the sensor-to-sky-patch angle
                if theta > el_angle:
                    # Indicate a bail out if the building is obstructing
                    hit_found = 1

            # Advance the ray stepper
            ray_step_ix = ray_step_ix + 1.0

            # Compute the new length
            distance = ray_step_ix * self.ray_step_size

            # Compute the new location
            next_loc = start + distance * slope

            # Tester for ray termination
            in_domain = (
                (next_loc.x > 0)
                and (next_loc.y > 0)
                and (next_loc.x < self.width)
                and (next_loc.y < self.length)
                and distance < self.max_ray_length
            )

        if hit_found == 0:
            distance = -1
        # TODO: this is bad!
        return distance

    @ti.kernel
    def load_3d_points(self):
        for sensor_ix in self.xyz_sensors:
            xyz_sensor = self.xyz_sensors[sensor_ix]
            parent_xy_sen = self.xy_sensors[
                self.xyz_sensors[sensor_ix].parent_sensor_id
            ]
            self.sensor_3d_points[sensor_ix].x = parent_xy_sen.loc.x
            self.sensor_3d_points[sensor_ix].y = xyz_sensor.height
            self.sensor_3d_points[sensor_ix].z = parent_xy_sen.loc.y
            self.sensor_3d_colors[sensor_ix].x = 0.5
            self.sensor_3d_colors[sensor_ix].y = ti.min(
                ti.max(
                    (xyz_sensor.rad - 650.0)
                    / (self.n_azimuths * self.n_elevations - 650),
                    0.0,
                ),
                1.0,
            )
            self.sensor_3d_colors[sensor_ix].z = 0.5

    @ti.kernel
    def load_3d_sensor_rays(self, sensor_ix: int):
        for az_ix, el_ix in ti.ndrange(self.n_azimuths, self.n_elevations):
            ray_ix = az_ix * self.n_elevations + el_ix
            # get the xyz sensors corresponding xy sensor
            parent_sensor_id = self.xyz_sensors[sensor_ix].parent_sensor_id
            parent_sensor = self.xy_sensors[parent_sensor_id]

            # get the xyz sensor's height
            xyz_sensor_height = self.xyz_sensors[sensor_ix].height

            el_angle = (
                el_ix * self.elevation_inc
            )  # TODO: precompute these? or store slopes?

            az_angle = (
                self.azimuth_inc * az_ix
                + self.edges[parent_sensor.parent_edge_id].az_start_angle
            )

            # Compute the ray's xy-plane slope
            dx = ti.cos(
                az_angle
            )  # TODO: precompute as a lookup in init based off of n_azimuths?
            dy = ti.sin(az_angle)
            slope = ti.Vector([dx, dy])

            # Get the ray's starting point
            start = parent_sensor.loc

            distance = self.trace_xyz_ray(start, slope, el_angle, xyz_sensor_height)

            if distance < 0:
                # hide the ray by setting the target to the source
                self.sensor_3d_rays[2 * ray_ix].x = parent_sensor.loc.x
                self.sensor_3d_rays[2 * ray_ix].y = xyz_sensor_height
                self.sensor_3d_rays[2 * ray_ix].z = parent_sensor.loc.y
            else:
                # the ray hit something
                self.sensor_3d_rays[2 * ray_ix].x = (
                    distance * slope.x + parent_sensor.loc.x
                )
                self.sensor_3d_rays[2 * ray_ix].y = (
                    xyz_sensor_height
                    + ti.tan(el_angle)
                    * ti.sqrt(slope.x * slope.x + slope.y * slope.y)
                    * distance
                )
                self.sensor_3d_rays[2 * ray_ix].z = (
                    distance * slope.y + parent_sensor.loc.y
                )

            self.sensor_3d_rays[2 * ray_ix + 1].x = parent_sensor.loc.x
            self.sensor_3d_rays[2 * ray_ix + 1].y = xyz_sensor_height
            self.sensor_3d_rays[2 * ray_ix + 1].z = parent_sensor.loc.y

    @ti.kernel
    def print_column_stats(self, xy_sensor: int):
        """
        Given an xy sensor, this function will print out the
        stats for each of the xyz sensors in the column above it.
        Useful for debugging.
        """
        print(f"\ncolumn for xy sensor {xy_sensor}")

        # Get the XY Sensor
        sensor = self.xy_sensors[xy_sensor]

        # Find where the stack of xyz sensors starts
        xyz_s = sensor.xyz_sensor_start_ix

        # Get the number of sensors above (from n floors)
        xyz_e = xyz_s + sensor.xyz_sensor_ct

        for xyz_sensor_ix in range(xyz_s, xyz_e):
            # Get the corresponding sensor
            sen = self.xyz_sensors[xyz_sensor_ix]
            # Get the radiation
            rad = sen.rad
            print(f"Floor {xyz_sensor_ix}: {rad} rad")
        ti.sync()

        # Check for hits for each individual ray
        for xyz_sensor_ix in range(xyz_s, xyz_e):
            sum = 0.0
            for az_ix, el_ix in ti.ndrange(self.n_azimuths, self.n_elevations):
                # For each ray, check if it's a hit
                if ti.is_active(self.xyz_view_root, [xyz_sensor_ix, az_ix, el_ix]) == 1:
                    sum = sum + 1
            print(f"Floor {xyz_sensor_ix}: {sum} hits")

    def get_sensor_hits_as_im(self, sensor_ix: int) -> ti.ScalarField:
        im = ti.field(float, shape=(2**self.depth, 2**self.depth))
        self.set_sensor_hits_im_kernel(im, sensor_ix=sensor_ix)

        return im

    def get_sensor_hits_as_pts(self, sensor_ix: int) -> ti.ScalarField:
        cur = ti.field(int, shape=())
        circs = ti.Vector.field(
            2, dtype=float, shape=self.xy_sensors[sensor_ix].hit_count
        )
        self.set_sensor_hits_pts_kernel(sensor_ix, pts=circs, cur=cur)
        return circs

    def get_colored_hit_map(self, sensor_ix: int):
        sensor_im = self.get_sensor_hits_as_im(sensor_ix)
        color_im = ti.Vector.field(
            3, dtype=float, shape=(2**self.depth, 2**self.depth)
        )
        self.combine_edge_and_sensor_hit_maps_kernel(
            sensor_im=sensor_im, color_im=color_im
        )

        return color_im

    def get_sensor_to_first_hit_rays(self, sensor_ix: int) -> ti.math.vec2:
        first_hit_points = ti.Vector.field(2, dtype=float, shape=self.n_azimuths + 1)
        indices = ti.field(int, shape=2 * self.n_azimuths)
        self.set_first_hit_points_kernel(
            sensor_ix=sensor_ix, pts=first_hit_points, indices=indices
        )
        return first_hit_points, indices

    @ti.kernel
    def set_first_hit_points_kernel(
        self, sensor_ix: int, pts: ti.template(), indices: ti.template()
    ):
        # TODO: ASSUMES POINTS ARE SORTED
        pts[self.n_azimuths] = self.xy_sensors[sensor_ix].loc
        for az_ix in range(self.n_azimuths):
            if self.hits[sensor_ix, az_ix].length() > 0:
                loc = self.hits[
                    sensor_ix, az_ix, 0
                ].centroid()  # TODO: Assumes a 1m grid spacing
                pts[az_ix] = loc
            else:
                az_angle = (
                    self.azimuth_inc * az_ix
                    + self.edges[
                        self.xy_sensors[sensor_ix].parent_edge_id
                    ].az_start_angle
                )
                dx = ti.cos(az_angle)  # TODO: precompute as a lookup
                dy = ti.sin(az_angle)  # TODO: precompute as a lookup
                slope = ti.Vector([dx, dy])
                pts[az_ix] = pts[self.n_azimuths] + slope * 500
            indices[az_ix * 2] = self.n_azimuths
            indices[az_ix * 2 + 1] = az_ix

    @ti.kernel
    def set_sensor_hits_pts_kernel(
        self, sensor_ix: int, pts: ti.template(), cur: ti.template()
    ):
        for az_ix in range(self.n_azimuths):
            for hit_ix in range(self.hits[sensor_ix, az_ix].length()):
                hit = self.hits[sensor_ix, az_ix, hit_ix]
                pts[
                    ti.atomic_add(cur[None], 1)
                ] = hit.centroid()  # TODO: Assumes a 1m grid spacing

    @ti.kernel
    def set_sensor_hits_im_kernel(self, im: ti.template(), sensor_ix: int):
        for az_ix in range(self.n_azimuths):
            for hit_ix in range(self.hits[sensor_ix, az_ix].length()):
                hit = self.hits[sensor_ix, az_ix, hit_ix]
                im[hit.loc_x_ix, hit.loc_y_ix] = 1

    @ti.kernel
    def combine_edge_and_sensor_hit_maps_kernel(
        self,
        sensor_im: ti.template(),
        color_im: ti.template(),
    ):
        for i, j in color_im:
            if ti.is_active(self.tree_leaves, [i, j]):
                color_im[i, j] = ti.Vector([1, 1, 1])
            if sensor_im[i, j] > 0:
                color_im[i, j] = ti.Vector([1, 0, 0])

    @property
    def tree_leaves(self):
        return self.levels[-1]

    @property
    def tree_root(self):
        return self.levels[0]


if __name__ == "__main__":
    import os
    from pathlib import Path

    fp = Path(os.path.abspath(os.path.dirname(__file__))) / "Braga_Baseline.zip"
    height_col = "height (m)"
    id_col = "id"

    # fp = Path(os.path.abspath(os.path.dirname(__file__))) / "sandySprings_Footprints.zip"
    # height_col = "BuildingHe"
    # id_col = "OBJECTID"

    tracer = Tracer(filepath=fp, height_col=height_col, id_col=id_col, node_width=1)
    ti.sync()
    tracer.print_column_stats(0)
    ti.sync()

    tracer.init_gui()
    tracer.render_scene()
    exit()

    ############
    # debug viz
    window = ti.ui.Window("GIS", (1024, 1024), pos=(50, 50))
    canv = window.get_canvas()
    ui = window.get_gui()

    # TODO: move edge points into class
    @ti.kernel
    def set_edge_verts_kernel(
        edge_starts: ti.template(), edge_ends: ti.template(), edge_verts: ti.template()
    ):
        for i in range(edge_starts.shape[0]):
            edge_verts[2 * i] = ti.Vector([edge_starts[i, 0], edge_starts[i, 1]])
            edge_verts[2 * i + 1] = ti.Vector([edge_ends[i, 0], edge_ends[i, 1]])

    @ti.kernel
    def set_edge_colors_kernel(edge_starts: ti.template(), edge_colors: ti.template()):
        for i in range(edge_starts.shape[0]):
            edge_colors[2 * i] = (
                ti.Vector([ti.random(), ti.random(), ti.random()]) * 0.5
            )
            edge_colors[2 * i + 1] = edge_colors[2 * i]

    @ti.kernel
    def zoom_pan_im_kernel(
        source_im: ti.template(), target_im: ti.template(), x_offset: int, y_offset: int
    ):
        for i, j in target_im:
            target_im[i, j] = source_im[i + x_offset, j + y_offset]

    # TODO: make a parent function which copies and zooms at the same time so the underlying points don't need to be reassembled.
    @ti.kernel
    def zoom_pan_pts_kernel(
        source_pts: ti.template(),
        zoom: float,
        x_offset: int,
        y_offset: int,
        zoom_base: float,
    ):
        for i in source_pts:
            source_pts[i] = (source_pts[i] - ti.Vector([x_offset, y_offset])) / (
                zoom * zoom_base
            )

    def zoom_pan_im(source_im, zoom: float, x_offset: int, y_offset: int):
        target_im = ti.Vector.field(
            3,
            float,
            shape=(int(source_im.shape[0] * zoom), int(source_im.shape[0] * zoom)),
        )
        zoom_pan_im_kernel(
            source_im=source_im,
            target_im=target_im,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        return target_im

    edge_ct = tracer.edge_starts.shape[0]
    borderline_verts = ti.Vector.field(2, dtype=float, shape=(2 * edge_ct))
    # borderline_colors = ti.Vector.field(3, dtype=float, shape=(2 * edge_ct))

    # set_edge_colors_kernel(tracer.edge_starts, borderline_colors)
    set_edge_verts_kernel(tracer.edge_starts, tracer.edge_ends, borderline_verts)

    sensor_ix = 0
    circs = tracer.get_sensor_hits_as_pts(sensor_ix)
    hit_lines, indices = tracer.get_sensor_to_first_hit_rays(sensor_ix)

    zoom = 1
    x_offset = 0
    y_offset = 0

    controls_changed = True
    while window.running:
        with ui.sub_window("Sensor selector", 0.1, 0.1, 0.8, 0.15):
            old_ix = sensor_ix
            sensor_ix = ui.slider_int(
                text="Sensor Index",
                old_value=sensor_ix,
                minimum=0,
                maximum=tracer.xy_sensors.shape[0],
            )
            if old_ix != sensor_ix:
                controls_changed = True

            old_zoom = zoom
            zoom = ui.slider_float(
                text="Zoom Level",
                old_value=zoom,
                minimum=0.0001,
                maximum=1,
            )
            if old_zoom != zoom:
                controls_changed = True

            old_x_offset = x_offset
            x_offset = ui.slider_int(
                text="X Offset",
                old_value=x_offset,
                minimum=0,
                maximum=int(
                    tracer.node_heights.shape[0] - zoom * tracer.node_heights.shape[0]
                ),
            )
            if old_x_offset != x_offset:
                controls_changed = True

            old_y_offset = y_offset
            y_offset = ui.slider_int(
                text="Y Offset",
                old_value=y_offset,
                minimum=0,
                maximum=int(
                    tracer.node_heights.shape[0] - zoom * tracer.node_heights.shape[0]
                ),
            )
            if old_y_offset != y_offset:
                controls_changed = True

            if controls_changed:
                circs = tracer.get_sensor_hits_as_pts(sensor_ix)
                hit_verts, _ = tracer.get_sensor_to_first_hit_rays(sensor_ix)
                set_edge_verts_kernel(
                    tracer.edge_starts,
                    tracer.edge_ends,
                    borderline_verts,
                )

                zoom_pan_pts_kernel(
                    source_pts=circs,
                    zoom=zoom,
                    zoom_base=tracer.node_heights.shape[0],
                    x_offset=x_offset,
                    y_offset=y_offset,
                )
                zoom_pan_pts_kernel(
                    source_pts=hit_verts,
                    zoom=zoom,
                    zoom_base=tracer.node_heights.shape[0],
                    x_offset=x_offset,
                    y_offset=y_offset,
                )
                zoom_pan_pts_kernel(
                    source_pts=borderline_verts,
                    zoom=zoom,
                    zoom_base=tracer.node_heights.shape[0],
                    x_offset=x_offset,
                    y_offset=y_offset,
                )
            controls_changed = False

        canv.lines(
            borderline_verts, 0.001, color=(1, 1, 1)
        )  # per_vertex_color=borderline_colors)
        canv.lines(hit_verts, 0.002, color=(1, 0, 0), indices=indices)
        canv.circles(circs, radius=0.002, color=(1, 0, 0))

        window.show()