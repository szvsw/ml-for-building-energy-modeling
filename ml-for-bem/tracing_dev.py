import logging

from os import PathLike
from typing import List, Union

import taichi as ti
import numpy as np
import geopandas as gpd
import pandas as pd

from shapely import Polygon, LineString, MultiPoint, Point

# ti.init(arch=ti.gpu, device_memory_fraction=0.7, kernel_profiler=True, debug=True)
ti.init(arch=ti.cpu, kernel_profiler=True)
# ti.init(arch=ti.gpu, device_memory_fraction=0.7, kernel_profiler=True)

logging.basicConfig()
logger = logging.getLogger("Radiation Analysis")
logger.setLevel(logging.INFO)

@ti.dataclass
class XYSensor:
    hit_count: int
    loc: ti.math.vec2
    parent_edge_id: int


@ti.dataclass
class Hit:
    loc_x_ix: ti.i16
    loc_y_ix: ti.i16
    height: float

    @ti.func
    def centroid(self) -> ti.math.vec2:
        return ti.Vector([self.loc_x_ix + 0.5, self.loc_y_ix + 0.5]) # TODO: assumes a bin spacing of 1m!


@ti.data_oriented
class Tracer:
    node_width: float  # meters
    sensor_inset: float  # meters
    sensor_spacing: float  # meters
    f2f_height: float  # floor-to-floor height, meters
    depth: int  # quadtree level count
    levels: List[ti.SNode]  # pointers to each 2x2 level of the quadtree
    node_heights: ti.ScalarField  # stores the height of each populated node in the quadtree

    # TODO: consider combining edges into a structfield with @ti.dataclass (requires a copy kernel to populate)
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
        sensor_spacing: float = 3,
        f2f_height: float = 3,
        convert_crs=False,
    ):
        # TODO: add meter conversion, better crs validation/conversion
        # store the bin size in meters
        assert (
            node_width == 1
        ), "Currently only supports dividing the space into 1 m node widths"
        self.node_width = node_width
        padding = 5 * node_width

        # store the sensor grid config
        self.sensor_inset = sensor_inset
        self.sensor_spacing = sensor_spacing
        self.f2f_height = f2f_height

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

        # self.gdf = self.gdf.loc[self.gdf.index.repeat(200)].reset_index()
        base_gdf = self.gdf.copy()
        tile_ct = 10
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
        self.levels = [ti.root.pointer(ti.ij, (2**self.depth, 2**self.depth))]
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
        self.n_azimuths = 24
        self.azimuth_inc = 2 * np.pi / self.n_azimuths
        self.sensor_root = ti.root.dense(ti.i, sensor_count)
        self.ray_root = self.sensor_root.dense(ti.j,  self.n_azimuths)
        self.hit_block = self.ray_root.dynamic(ti.k, 1024, chunk_size=32)
        self.hits = Hit.field()
        self.hit_block.place(self.hits)

        # Init xy sensor locations
        logger.info("Initializing xy-plane sensors...")
        self.xy_sensors = XYSensor.field() # TODO: Why do I have to place hit_block first for compilation to work?
        self.sensor_root.place(self.xy_sensors)
        self.init_xy_sensors()
        ti.sync()

        # Ray trace in xy plane
        logger.info("XY tracing...")
        self.xy_trace()
        ti.sync()
        logger.info("XY tracing complete.")

        ti.profiler.print_kernel_profiler_info()

    def extract_flat_edge_list(self):
        """
        Extracts all edges to a flattened list
        # TODO: figure out if there is a vectorized manner instead of having to do it bldg by bldg
        """
        starts = []
        ends = []
        run_rises = []
        heights = []
        normals = []
        building_ids = []
        normal_fails = 0
        for i, _geom in enumerate(self.gdf.geometry):
            # Manual expldoe of geometry for multiploygon handling
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
                heights.append(np.ones(points.shape[0]) * self.gdf[self.height_col][i])
                normals.append(normal)

        starts = np.vstack(starts)
        ends = np.vstack(ends)
        run_rises = np.vstack(run_rises)
        normals = np.vstack(normals)
        heights = np.concatenate(heights)

        lengths = np.linalg.norm(starts - ends, axis=1)
        sensor_counts = (lengths - 2 * self.sensor_inset) / self.sensor_spacing
        sensor_counts = np.floor(np.where(sensor_counts >= 1, sensor_counts + 1, 0))
        sensor_ends = np.cumsum(sensor_counts)
        sensor_starts = np.roll(sensor_ends, shift=1)
        sensor_starts[0] = 0
        sensor_parent_ix = np.repeat(
            np.arange(sensor_counts.shape[0]), sensor_counts.astype(int)
        )

        # Create the fields
        self.edge_starts = ti.field(float, shape=starts.shape)
        self.edge_ends = ti.field(float, shape=ends.shape)
        self.edge_slopes = ti.field(float, shape=run_rises.shape)
        self.edge_heights = ti.field(float, shape=heights.shape)
        self.edge_normals = ti.field(float, shape=normals.shape)
        self.edge_sensor_starts = ti.field(int, shape=sensor_starts.shape)
        self.edge_sensor_ends = ti.field(int, shape=sensor_ends.shape)
        self.edge_sensor_counts = ti.field(int, shape=sensor_counts.shape)
        self.edge_sensor_parent_ix = ti.field(int, shape=sensor_parent_ix.shape)

        # Copy the numpy data over
        self.edge_starts.from_numpy(starts)
        self.edge_ends.from_numpy(ends)
        self.edge_slopes.from_numpy(run_rises)
        self.edge_heights.from_numpy(heights)
        self.edge_normals.from_numpy(normals)
        self.edge_sensor_starts.from_numpy(sensor_starts)
        self.edge_sensor_ends.from_numpy(sensor_ends)
        self.edge_sensor_counts.from_numpy(sensor_counts)
        self.edge_sensor_parent_ix.from_numpy(sensor_parent_ix)

    @ti.kernel
    def add_edges_to_tree(self):
        """
        This function determines where each line crosses a node threshold and 
        updates that node's height accordingly.
        """
        for edge_ix in self.edge_heights:
            # TODO: Update if edges switch to a dataclass representation
            # extract the endpoints/data
            x0 = self.edge_starts[edge_ix, 0]
            y0 = self.edge_starts[edge_ix, 1]
            x1 = self.edge_ends[edge_ix, 0]
            y1 = self.edge_ends[edge_ix, 1]
            h = self.edge_heights[edge_ix]

            # compute slope
            slope = (y1 - y0) / (x1 - x0)  # TODO: handle vert/hor lines

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
            self.xy_sensors[sensor_ix].parent_edge_id = parent_id

            # copy the parent slope over
            slope = ti.Vector(
                [self.edge_slopes[parent_id, 0], self.edge_slopes[parent_id, 1]]
            )  # TODO: the slope field should be a vector2 field)

            # Determine the inset edge gap for the sensor
            start_gap = slope * self.sensor_inset

            # Determine which sensor this is along a the parent edge 
            gap_ct = sensor_ix - self.edge_sensor_starts[parent_id]

            # compute the distance from the edge start vertex
            distance = start_gap + gap_ct * slope * self.sensor_spacing
            
            # Copy the parent edge start vertex over
            start_loc = ti.Vector(
                [self.edge_starts[parent_id, 0], self.edge_starts[parent_id, 1]]
            )  # TODO: the point field should be a vector2 field)

            # Copy the parent edge normal over
            normal = ti.Vector(
                [self.edge_normals[parent_id, 0], self.edge_normals[parent_id, 1]]
            )  # TODO: the point field should be a vector2 field)

            # Set the new location by moving along edge the appropriate amount
            # and then 1.5m away from the wall following the normal
            self.xy_sensors[sensor_ix].loc = (
                start_loc + distance + normal * 1.5
            )   # TODO: make this a class attr

    @ti.kernel
    def xy_trace(self):
        max_ray_length = 400.0 # TODO: assumes max radius for rays
        dcur = 1.0 # TODO: assumes 1m ray hops, will cause duplicate collisions
        n_curs = ti.floor(max_ray_length / dcur, dtype=int)
        # TODO: this version (i.e. the non divergent version which does not use nested for loop) may cause 
        # overflow in ndrange if the product is too large
        for sensor_ix, az_ix, cur_ix in ti.ndrange(self.xy_sensors.shape[0],self.n_azimuths, n_curs):
            # Compute the rays's azimuth angle
            sensor = self.xy_sensors[sensor_ix]

            az_angle = self.azimuth_inc * az_ix 

            # Compute the ray's xy-plane slope
            dx = ti.cos(az_angle)  # TODO: precompute as a lookup in init based off of n_azimuths?
            dy = ti.sin(az_angle)  
            slope = ti.Vector([dx, dy])

            # Get the ray's starting point
            start = sensor.loc

            # Length of ray to check
            # TODO: store this
            l = dcur * cur_ix

            # Initializing the next location to check
            next_loc = start + cur_ix*dcur * slope
            
            # Tester for ray termination
            in_domain = (
                (next_loc.x > 0)
                and (next_loc.y > 0)
                and (next_loc.x < self.width)
                and (next_loc.y < self.length)
            )
            if in_domain:
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
                        )  # TODO: assumes a  grid spacing = 1
                    )
                    self.xy_sensors[sensor_ix].hit_count += 1

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
        first_hit_points = ti.Vector.field(
            2, dtype=float, shape=self.n_azimuths+1
        )
        indices = ti.field(int, shape=2*self.n_azimuths)
        self.set_first_hit_points_kernel(sensor_ix=sensor_ix, pts=first_hit_points, indices=indices)
        return first_hit_points, indices
    
    @ti.kernel
    def set_first_hit_points_kernel(self, sensor_ix: int, pts: ti.template(), indices: ti.template()):
        # TODO: ASSUMES POINTS ARE SORTED
        pts[self.n_azimuths] = self.xy_sensors[sensor_ix].loc
        for az_ix in range(self.n_azimuths):
            if self.hits[sensor_ix, az_ix].length() > 0:
                loc = self.hits[sensor_ix, az_ix, 0].centroid() # TODO: Assumes a 1m grid spacing
                pts[az_ix] = loc
            else:
                az_angle = self.azimuth_inc * az_ix
                dx = ti.cos(az_angle)  # TODO: precompute as a lookup
                dy = ti.sin(az_angle)  # TODO: precompute as a lookup
                slope = ti.Vector([dx, dy])
                pts[az_ix] = pts[self.n_azimuths] + slope*500
            indices[az_ix*2] = self.n_azimuths
            indices[az_ix*2+1] = az_ix


    @ti.kernel
    def set_sensor_hits_pts_kernel(
        self, sensor_ix: int, pts: ti.template(), cur: ti.template()
    ):
        for az_ix in range(self.n_azimuths):
            for hit_ix in range(self.hits[sensor_ix, az_ix].length()):
                hit = self.hits[sensor_ix, az_ix, hit_ix]
                pts[ti.atomic_add(cur[None], 1)] = hit.centroid() # TODO: Assumes a 1m grid spacing
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
    # print(tracer.node_heights.shape)

    #

    ############
    # debug viz
    window = ti.ui.Window("GIS", (512, 512), pos=(100, 100))
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


    zoom = 0.5
    x_offset = 0
    y_offset = 0

    controls_changed = True
    while window.running:
        with ui.sub_window("Sensor selector", 0.1, 0.1, 0.3, 0.3):
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
            borderline_verts, 0.005, color=(1, 1, 1)
        )  # per_vertex_color=borderline_colors)
        canv.lines(
            hit_verts, 0.005, color=(1, 0,0), indices=indices
        )  # per_vertex_color=borderline_colors)
        canv.circles(circs, radius=0.005, color=(1, 0, 0))

        window.show()
