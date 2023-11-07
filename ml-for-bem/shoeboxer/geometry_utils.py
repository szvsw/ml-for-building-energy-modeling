import logging
import math

import numpy as np

from shoeboxer.shoebox_config import ShoeboxConfiguration

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

BASE_PERIM_DEPTH = 5  # TODO make dynamic
BASE_CORE_DEPTH = 10
BASE_WIDTH = 3
BASE_HEIGHT = 4
BASE_ADIABATIC_FRAC = 0.5
ZONE_DEPTH_LIMIT = 0.3
SURF_DEPTH_LIMIT = 0.1


def scale_shoebox(
    sb,
    width,
    height,
    perim_depth,
    core_depth,
):
    logger.info(f"New perimeter depth: {perim_depth}")
    logger.info(f"New core depth: {core_depth}")

    if core_depth < ZONE_DEPTH_LIMIT or perim_depth < ZONE_DEPTH_LIMIT:
        logger.error(f"Depth is too small. Dropping shoebox.")
        raise ValueError("Depth is too small.")

    perim_idxs = [
        i
        for i, x in enumerate(sb["BuildingSurface:Detailed"].keys())
        if "PERIM" in x.upper()
    ]
    core_idxs = [
        i
        for i, x in enumerate(sb["BuildingSurface:Detailed"].keys())
        if "CORE" in x.upper()
    ]

    window_coords = get_window_coords(
        sb["FenestrationSurface:Detailed"]["Perimeter:f0:w0"]
    )
    window_coords = np.expand_dims(window_coords, axis=0)

    all_coords = get_all_coords(sb["BuildingSurface:Detailed"])

    # Scale width
    new_coords = scale_1D(all_coords, width / BASE_WIDTH, axis=0)
    # Scale height
    new_coords = scale_1D(new_coords, height / BASE_HEIGHT, axis=2)

    # Recenter the window on the new x-centerline
    original_window_xcenter = BASE_WIDTH / 2
    new_window_xcenter = width / 2
    window_move_distance = new_window_xcenter - original_window_xcenter
    window_coords = move_1D(window_coords, window_move_distance, axis=0)

    # Scale window width, keeping the x-centerline anchored
    window_coords = scale_1D(
        window_coords, width / BASE_WIDTH, axis=0, anchor=new_window_xcenter
    )

    # Recenter the window on the new z-centerline
    old_window_zcenter = BASE_HEIGHT / 2
    new_window_zcenter = height / 2
    window_move_distance = new_window_zcenter - old_window_zcenter
    window_coords = move_1D(window_coords, window_move_distance, axis=2)
    # Scale window height, keeping the z-centerline anchored
    window_coords = scale_1D(
        window_coords, height / BASE_HEIGHT, axis=2, anchor=new_window_zcenter
    )

    # Scale perimeter
    new_coords[perim_idxs, :, :] = scale_1D(
        new_coords[perim_idxs, :, :], perim_depth / BASE_PERIM_DEPTH, axis=1
    )
    # Move core
    logger.info(
        f"Shifting core origin to perimeter edge; moving {perim_depth - BASE_PERIM_DEPTH} m in y axis"
    )
    new_coords[core_idxs, :, :] = move_1D(
        new_coords[core_idxs, :, :], perim_depth - BASE_PERIM_DEPTH, axis=1
    )
    # Scale core
    new_coords[core_idxs, :, :] = scale_1D(
        new_coords[core_idxs, :, :], core_depth / BASE_CORE_DEPTH, axis=1
    )
    sb["BuildingSurface:Detailed"] = replace_all_coords(
        sb["BuildingSurface:Detailed"], new_coords
    )
    sb["FenestrationSurface:Detailed"]["Perimeter:f0:w0"] = replace_window_coords(
        sb["FenestrationSurface:Detailed"]["Perimeter:f0:w0"], window_coords[0, :, :]
    )
    return sb


def zone_depth(zone_coords):
    y_coords = zone_coords[:, :, 1].flatten()
    return max(y_coords) - min(y_coords)


def set_adiabatic_surfaces(sb, shoebox_config: ShoeboxConfiguration):
    logger.info("Updating adiabatic surfaces.")

    perim_depth = shoebox_config.perim_depth
    core_depth = shoebox_config.core_depth
    roof_2_footprint = shoebox_config.roof_2_footprint
    ground_2_footprint = shoebox_config.ground_2_footprint

    # SET PARTITION BETWEEN ZONES TO AN ADIABATIC SURFACE
    if shoebox_config.adiabatic_partition_flag == 1:
        for x, y in sb["BuildingSurface:Detailed"].items():
            if y["outside_boundary_condition"].upper() == "SURFACE":
                logger.debug(f"Setting {x} as adiabatic.")
                y.pop("outside_boundary_condition_object")
                y["outside_boundary_condition"] = "Adiabatic"

    perim_roof_exposed_depth = (
        roof_2_footprint * perim_depth
    )  # TODO: check that this is true
    perim_ground_exposed_depth = ground_2_footprint * perim_depth
    core_roof_exposed_depth = roof_2_footprint * core_depth
    core_ground_exposed_depth = ground_2_footprint * core_depth
    logger.debug(
        f"Core depth of adiabatic and non-adiabatic: {core_roof_exposed_depth}, {core_depth-core_roof_exposed_depth}"
    )
    perim_roof_lookups = [
        (i, x)
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if (
            "PERIM" in x.upper()
            and (
                "CEILING" in y["surface_type"].upper()
                or "ROOF" in y["surface_type"].upper()
            )
        )
    ]
    core_roof_lookups = [
        (i, x)
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if (
            "CORE" in x.upper()
            and (
                "CEILING" in y["surface_type"].upper()
                or "ROOF" in y["surface_type"].upper()
            )
        )
    ]
    perim_ground_lookups = [
        (i, x)
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if ("PERIM" in x.upper() and "FLOOR" in y["surface_type"].upper())
    ]
    core_ground_lookups = [
        (i, x)
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if ("CORE" in x.upper() and "FLOOR" in y["surface_type"].upper())
    ]
    perim_roof_idxs = [i[0] for i in perim_roof_lookups]
    perim_roof_names = [x[1] for x in perim_roof_lookups]
    perim_ground_idxs = [i[0] for i in perim_ground_lookups]
    perim_ground_names = [x[1] for x in perim_ground_lookups]
    core_roof_idxs = [i[0] for i in core_roof_lookups]
    core_roof_names = [x[1] for x in core_roof_lookups]
    core_ground_idxs = [i[0] for i in core_ground_lookups]
    core_ground_names = [x[1] for x in core_ground_lookups]

    uniform_exterior_condition_roof_c = False
    uniform_exterior_condition_roof_p = False
    uniform_exterior_condition_ground_c = False
    uniform_exterior_condition_ground_p = False

    if perim_roof_exposed_depth < SURF_DEPTH_LIMIT:
        # make adiabatic
        logger.warning(f"Perim exposed depth is too small. Making adiabatic.")
        for name in perim_roof_names:
            data = {
                "construction_name": "Interior Slab",
                "outside_boundary_condition": "Adiabatic",
                "sun_exposure": "NoSun",
                "surface_type": "Ceiling",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_roof_p = True

    if core_roof_exposed_depth < SURF_DEPTH_LIMIT:
        # make outdoor
        logger.warning(f"Core exposed depth is too small. Making adiabatic.")
        for name in core_roof_names:
            data = {
                "construction_name": "Interior Slab",
                "outside_boundary_condition": "Adiabatic",
                "sun_exposure": "NoSun",
                "surface_type": "Ceiling",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_roof_c = True

    if perim_ground_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(f"Perim ground exposed depth is too small. Making adiabatic.")
        for name in perim_ground_names:
            data = {
                "construction_name": "Interior Slab_FLIPPED",
                "outside_boundary_condition": "Adiabatic",
                "sun_exposure": "NoSun",
                "surface_type": "Floor",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_ground_p = True

    if core_ground_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(f"Core ground exposed depth is too small. Making adiabatic.")
        for name in core_ground_names:
            data = {
                "construction_name": "Interior Slab_FLIPPED",
                "outside_boundary_condition": "Adiabatic",
                "sun_exposure": "NoSun",
                "surface_type": "Floor",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_ground_c = True

    if perim_depth - perim_roof_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(f"Perim roof adiabatic depth is too small. Dropping adiabatic.")
        for name in perim_roof_names:
            data = {
                "construction_name": "Exterior Roof",
                "outside_boundary_condition": "Outdoors",
                "sun_exposure": "SunExposed",
                "surface_type": "Roof",
                "wind_exposure": "WindExposed",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_roof_p = True

    if core_depth - core_roof_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(f"Core roof adiabatic depth is too small. Dropping adiabatic.")
        for name in core_roof_names:
            data = {
                "construction_name": "Exterior Roof",
                "outside_boundary_condition": "Outdoors",
                "sun_exposure": "SunExposed",
                "surface_type": "Roof",
                "wind_exposure": "WindExposed",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_roof_c = True

    if perim_depth - perim_ground_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(
            f"Perim adiabatic ground depth is too small. Dropping adiabatic."
        )
        for name in perim_ground_names:
            data = {
                "construction_name": "Ground Slab",
                "outside_boundary_condition": "Ground",
                "sun_exposure": "NoSun",
                "surface_type": "Floor",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_ground_p = True

    if core_depth - core_ground_exposed_depth < SURF_DEPTH_LIMIT:
        logger.warning(f"Core adiabatic ground depth is too small. Dropping adiabatic.")
        for name in core_ground_names:
            data = {
                "construction_name": "Ground Slab",
                "outside_boundary_condition": "Ground",
                "sun_exposure": "NoSun",
                "surface_type": "Floor",
                "wind_exposure": "NoWind",
            }
            sb["BuildingSurface:Detailed"][name].update(data)
        uniform_exterior_condition_ground_c = True

    # SHIFTING OF ADIABATIC LINE
    if not all(
        (
            uniform_exterior_condition_roof_p,
            uniform_exterior_condition_ground_p,
            uniform_exterior_condition_roof_c,
            uniform_exterior_condition_ground_c,
        )
    ):
        logger.debug("Shifting adiabatic lines")
        all_coords = get_all_coords(sb["BuildingSurface:Detailed"])
        core_y_origin = min(all_coords[core_roof_idxs, :, 1].flatten())
        original_perim_adiabatic_y_coord = (
            zone_depth(all_coords[perim_roof_idxs]) * BASE_ADIABATIC_FRAC
        )
        original_core_adiabatic_y_coord = (
            core_y_origin + zone_depth(all_coords[core_roof_idxs]) * BASE_ADIABATIC_FRAC
        )

        if not all(
            (uniform_exterior_condition_roof_p, uniform_exterior_condition_ground_p)
        ):
            if not uniform_exterior_condition_roof_p:
                # move perim adiabatic line for roof
                pr = all_coords[perim_roof_idxs]
                mask = np.abs((pr - original_perim_adiabatic_y_coord)) < 0.0001
                mask[:, :, 0] = False  # we are only interested in y coords
                mask[:, :, 2] = False
                pr[mask] = perim_roof_exposed_depth
                all_coords[perim_roof_idxs] = pr
            if not uniform_exterior_condition_ground_p:
                # move perim adiabatic line for floor
                pf = all_coords[perim_ground_idxs]
                mask = np.abs((pf - original_perim_adiabatic_y_coord)) < 0.0001
                mask[:, :, 0] = False  # we are only interested in y coords
                mask[:, :, 2] = False
                pf[mask] = perim_ground_exposed_depth
                all_coords[perim_ground_idxs] = pf

        if not all(
            (uniform_exterior_condition_roof_c, uniform_exterior_condition_ground_c)
        ):
            logger.debug("Shifting core adiabatic lines")
            if not uniform_exterior_condition_roof_c:
                # move core adiabatic line for roof
                cr = all_coords[core_roof_idxs]
                mask = np.abs((cr - original_core_adiabatic_y_coord)) < 0.0001
                mask[:, :, 0] = False  # we are only interested in y coords
                mask[:, :, 2] = False
                cr[mask] = core_roof_exposed_depth + core_y_origin
                all_coords[core_roof_idxs] = cr
            if not uniform_exterior_condition_ground_c:
                logger.debug("Shifting core ground adiabatic lines")
                # move core adiabatic line for floor
                cf = all_coords[core_ground_idxs]
                mask = np.abs((cf - original_core_adiabatic_y_coord)) < 0.0001
                mask[:, :, 0] = False  # we are only interested in y coords
                mask[:, :, 2] = False
                cf[mask] = core_ground_exposed_depth + core_y_origin
                all_coords[core_ground_idxs] = cf

        sb["BuildingSurface:Detailed"] = replace_all_coords(
            sb["BuildingSurface:Detailed"], all_coords
        )

    return sb


def vertices_to_coords(vertices):
    coords = []
    for v in vertices:
        coords.append(list(v.values()))
    return coords


def get_coords(buildingsurf):
    coords_all = []
    for vertex in buildingsurf["vertices"]:
        coords_all.append(list(vertex.values()))
    coords_all = np.array(coords_all)
    return coords_all


def replace_coords(buildingsurf, coords):
    for i, vertex in enumerate(buildingsurf["vertices"]):
        for j, a in enumerate(["x", "y", "z"]):
            vertex[f"vertex_{a}_coordinate"] = coords[i, j]


def get_all_coords(surfaces):
    all_coords = []
    for name, surface in surfaces.items():
        # change height
        coords = get_coords(surface)
        all_coords.append(coords)
    all_coords = np.array(all_coords)
    return all_coords


def replace_all_coords(surfaces, all_coords):
    # replace coordinates
    for coords, (name, surf) in zip(all_coords, surfaces.items()):
        surf = replace_coords(surf, coords)
    return surfaces


def scale_1D(all_coords, scale, axis, anchor=None):
    if anchor is None:
        anchor = min(all_coords[:, :, axis].flatten())

    move_1D(all_coords, -anchor, axis)
    for coords in all_coords:
        max_dim = max(coords[:, axis].flatten())
        coords[:, axis] *= scale
    move_1D(all_coords, anchor, axis)
    return all_coords


def move_1D(coords, distance, axis):
    try:
        coords[:, :, axis] = coords[:, :, axis] + distance
    except:
        coords[:, axis] = coords[:, axis] + distance
    return coords


def remove_surf(sb, surface_to_remove, all_coords=None):
    logger.info("REMOVING SURFACE")
    # get index of surface to remove
    keys = sb["BuildingSurface:Detailed"].keys()
    idx = keys.index(surface_to_remove)

    if all_coords is None:
        all_coords = get_all_coords(sb["BuildingSurface:Detailed"])
    logger.debug(all_coords.shape)
    normals = get_normals(all_coords)
    logger.debug(normals)
    working_normal = normals[idx]

    logger.debug(
        f"Removing {surface_to_remove} from {len(sb['BuildingSurface:Detailed'])} surfaces"
    )
    del sb["BuildingSurface:Detailed"][surface_to_remove]
    # drop from normals list and all_coords list
    normals = normals.pop(idx)
    all_coords = np.delete(all_coords, idx, 0)
    logger.debug(normals)
    logger.debug(all_coords.shape)
    logger.debug(f"Now, {len(sb['BuildingSurface:Detailed'])} surfaces")

    for i, normal in enumerate(normals):
        if np.cross(normal, working_normal) == 0:
            logger.debug(
                f"{sb['BuildingSurface:Detailed'][keys[i]]} is a normal surface."
            )
    # for name, surface_data in sb["BuildingSurface:Detailed"].items():
    # Get all parallel surfaces


def plane_normal(points):
    if len(points) < 3:
        raise ValueError("At least three points are required to calculate the normal.")

    v1 = np.array(points[1]) - np.array(points[0])
    v2 = np.array(points[2]) - np.array(points[0])

    # Calculate the cross product of v1 and v2 to get the normal vector
    normal = np.cross(v1, v2)

    # Normalize the normal vector to get a unit vector
    normal /= np.linalg.norm(normal)

    return normal


def get_normals(coords):
    normals = []
    for points in coords:
        normals.append(plane_normal(points))
    return normals


def get_window_coords(window):
    coords_all = []
    for i in range(1, 5):
        coords = []
        for a in ["x", "y", "z"]:
            coords.append(float(window[f"vertex_{i}_{a}_coordinate"]))
        coords_all.append(coords)
    return np.array(coords_all)


def replace_window_coords(window, coords):
    for i in range(4):
        for j, a in enumerate(["x", "y", "z"]):
            window[f"vertex_{i+1}_{a}_coordinate"] = coords[i][j]
    return window


def update_wwr(epjson, wwr_frac):
    if wwr_frac < 0.01:
        # Remove all windows
        epjson["FenestrationSurface:Detailed"] = {}
    else:
        for name, window in epjson["FenestrationSurface:Detailed"].items():
            coords_all = get_window_coords(window)
            # Check current wwr
            max_h = max(coords_all[:, 2])
            min_h = min(coords_all[:, 2])
            curr_window_h = max_h - min_h
            all_coords = get_all_coords(epjson["BuildingSurface:Detailed"])
            curr_h = max(all_coords[:, :, 2].flatten())
            logger.debug(
                f"Current window to wall ratio is {curr_window_h/curr_h} with height {curr_window_h}"
            )
            # Scale window to needed height - assuming window spans entire width of wall
            new_window_h = wwr_frac * curr_h
            logger.debug(f"New window height will be {new_window_h}")
            change = (
                new_window_h - curr_window_h
            ) / 2  # negative if new window is smaller
            # move max down
            coords_all[:, 2][coords_all[:, 2] == max_h] = max_h + change
            # move min up
            coords_all[:, 2][coords_all[:, 2] == min_h] = min_h - change
            # replace coordinates
            window = replace_window_coords(window, coords_all)
    return epjson


def build_shading(sb_json, angles, radius=10, override=True):  # TODO: is 12 too much?
    """
    Adds shading objects to a shoebox idf according to a list of heights (m)
    of equal width divided between 180 degrees at a given radius (m).
    Using geomeppy def add_shading_block(*args, **kwargs)
    """
    heights = [radius * math.tan(a) for a in angles]
    logger.info(f"Maximum shading point is {max(heights)}")

    if override:
        sb_json = remove_shading_surfaces(sb_json)

    # base point is at central bottom of shoebox - just in case shoebox is not centered at 0,0
    zone_names = [x["zone_name"] for x in sb_json["ZoneList"]["AllZones"]["zones"]]
    all_surfaces = sb_json["BuildingSurface:Detailed"]
    for surface_name, surface_data in all_surfaces.items():
        if (
            surface_data["surface_type"].upper() == "WALL"
            and surface_data["outside_boundary_condition"].upper() == "OUTDOORS"
        ):
            # coords = surface.coords
            coords = vertices_to_coords(surface_data["vertices"])
            facade_base_coords = []
            for coord in coords:
                if coord[2] == 0:
                    facade_base_coords.append(coord[:2])
    facade_base_coords = np.array(facade_base_coords)
    if len(facade_base_coords) > 2:
        raise TypeError("Multiple exterior walls found!")
    center_pt = np.mean(facade_base_coords, axis=0)
    logger.info(f"Placing shading around center point {center_pt}")

    # Get first point of shading
    vect = np.min(facade_base_coords, axis=0) - center_pt
    v_norm = vect / np.linalg.norm(vect)
    p_0 = v_norm * radius

    # make radial array
    count = len(heights)
    theta = math.pi / count

    rot_matrix = [
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ]

    shading_coords = np.array([p_0])
    pt = np.reshape(p_0, (2, 1))
    for h in heights:
        pt = np.matmul(rot_matrix, pt)
        shading_coords = np.append(shading_coords, np.reshape(pt, (1, 2)), axis=0)
    shading_coords[:, 0] += center_pt[0]
    shading_coords[:, 1] += center_pt[1]
    shading_epbunches = {}
    for i, h in enumerate(heights):
        if h > 0:
            name = f"SHADER_{i}"
            base_coords = shading_coords[i : i + 2, :]
            coords = []
            coords.append((base_coords[0, 0], base_coords[0, 1], h))
            coords.append((base_coords[0, 0], base_coords[0, 1], 0.0))
            coords.append((base_coords[1, 0], base_coords[1, 1], 0.0))
            coords.append((base_coords[1, 0], base_coords[1, 1], h))
            shading_epbunch = dict(
                # name=name,
                number_of_vertices=4,
                vertices=[
                    {
                        f"vertex_{j}_coordinate": c[k]
                        for k, j in enumerate(["x", "y", "z"])
                    }
                    for c in coords
                ],
            )
            shading_epbunches[name] = shading_epbunch
    sb_json["Shading:Building:Detailed"] = shading_epbunches
    return sb_json


def remove_shading_surfaces(sb_json):
    """
    Used to override existing shading surfces
    """
    logger.info("Removing all existing shading")
    sb_json["Shading:Building:Detailed"] = {}
    return sb_json


# def build_shading(sb, angles, radius=10, override=True):
#     """
#     Adds shading objects to a shoebox idf according to a list of heights (m)
#     of equal width divided between 180 degrees at a given radius (m).
#     Using geomeppy def add_shading_block(*args, **kwargs)
#     """
#     heights = [radius * math.tan(a) for a in angles]
#     logger.info(f"Maximum shading point is {max(heights)}")

#     if override:
#         remove_shading_surfaces(sb)

#     # base point is at central bottom of shoebox - just in case shoebox is not centered at 0,0
#     zones = sb.idfobjects["ZONE"]
#     for zone in zones:
#         for surface in zone.zonesurfaces:
#             if (
#                 surface.Surface_Type.upper() == "WALL"
#                 and surface.Outside_Boundary_Condition.upper() == "OUTDOORS"
#             ):
#                 coords = surface.coords
#                 facade_base_coords = []
#                 for coord in coords:
#                     if coord[2] == 0:
#                         facade_base_coords.append(coord[:2])
#     facade_base_coords = np.array(facade_base_coords)
#     if len(facade_base_coords) > 2:
#         raise TypeError("Multiple exterior walls found!")
#     center_pt = np.mean(facade_base_coords, axis=0)
#     logger.info(f"Placing shading around center point {center_pt}")

#     # Get first point of shading
#     vect = np.min(facade_base_coords, axis=0) - center_pt
#     v_norm = vect / np.linalg.norm(vect)
#     p_0 = v_norm * radius

#     # make radial array
#     count = len(heights)
#     theta = math.pi / count

#     rot_matrix = [
#         [math.cos(theta), -math.sin(theta)],
#         [math.sin(theta), math.cos(theta)],
#     ]

#     shading_coords = np.array([p_0])
#     pt = np.reshape(p_0, (2, 1))
#     for h in heights:
#         pt = np.matmul(rot_matrix, pt)
#         shading_coords = np.append(shading_coords, np.reshape(pt, (1, 2)), axis=0)
#     shading_coords[:, 0] += center_pt[0]
#     shading_coords[:, 1] += center_pt[1]

#     for i, h in enumerate(heights):
#         name = f"SHADER_{i}"
#         base_coords = shading_coords[i : i + 2, :]
#         coords = []
#         coords.append((base_coords[0, 0], base_coords[0, 1], h))
#         coords.append((base_coords[0, 0], base_coords[0, 1], 0.0))
#         coords.append((base_coords[1, 0], base_coords[1, 1], 0.0))
#         coords.append((base_coords[1, 0], base_coords[1, 1], h))
#         shading_epbunch = sb.newidfobject(
#             key="Shading:Building:Detailed".upper(),
#             Name=name,
#             Number_of_Vertices=4,
#         )
#         for j in coords:
#             shading_epbunch.fieldvalues.extend(j)
#     return sb


# def remove_shading_surfaces(sb):
#     """
#     Used to override existing shading surfces
#     """
#     for surface in sb.getshadingsurfaces():
#         sb.removeidfobject(surface)
