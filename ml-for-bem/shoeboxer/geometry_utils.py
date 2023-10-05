import math
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_PERIM_DEPTH = 5  # TODO make dynamic
BASE_CORE_DEPTH = 10
BASE_WIDTH = 3
BASE_HEIGHT = 4
BASE_ADIABATIC_FRAC = 0.5


def scale_shoebox(
    sb,
    width,
    height,
    floor_2_facade,
    core_2_perim,
):
    perim_depth = floor_2_facade * height
    core_depth = core_2_perim * perim_depth
    logger.info(f"New perimeter depth: {perim_depth}")
    logger.info(f"New core depth: {core_depth}")

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
    # Scale window width
    window_coords = scale_1D(window_coords, width / BASE_WIDTH, axis=0)
    # Scale height
    new_coords = scale_1D(new_coords, height / BASE_HEIGHT, axis=2)
    # Move window centerline
    window_coords = move_1D(window_coords, (height - BASE_HEIGHT) / 4, axis=2)
    # Scale window height
    window_coords = scale_1D(window_coords, height / BASE_HEIGHT, axis=2)
    # Scale perimeter
    new_coords[perim_idxs, :, :] = scale_1D(
        new_coords[perim_idxs, :, :], perim_depth / BASE_PERIM_DEPTH, axis=1
    )
    # Move core
    logger.info(f"Shifting core to perimeter {perim_depth - BASE_PERIM_DEPTH} m")
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


def set_adiabatic_surfaces(
    sb,
    height,
    floor_2_facade,
    core_2_perim,
    roof_2_footprint,
    ground_2_footprint,
):
    # TODO: remove if zero
    logger.info("Updating adiabatic surfaces.")
    perim_depth = floor_2_facade * height
    core_depth = core_2_perim * perim_depth

    perim_roof_depth = roof_2_footprint * perim_depth
    perim_ground_depth = ground_2_footprint * perim_depth
    core_roof_depth = roof_2_footprint * core_depth
    core_ground_depth = ground_2_footprint * core_depth

    perim_roof_idxs = [
        i
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if (
            "PERIM" in x.upper()
            and (
                "CEILING" in y["surface_type"].upper()
                or "ROOF" in y["surface_type"].upper()
            )
        )
    ]
    core_roof_idxs = [
        i
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if (
            "CORE" in x.upper()
            and (
                "CEILING" in y["surface_type"].upper()
                or "ROOF" in y["surface_type"].upper()
            )
        )
    ]
    perim_ground_idxs = [
        i
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if ("PERIM" in x.upper() and "FLOOR" in y["surface_type"].upper())
    ]
    core_ground_idxs = [
        i
        for i, (x, y) in enumerate(sb["BuildingSurface:Detailed"].items())
        if ("CORE" in x.upper() and "FLOOR" in y["surface_type"].upper())
    ]

    all_coords = get_all_coords(sb["BuildingSurface:Detailed"])

    old_roof_perim_depth = old_ground_perim_depth = (
        zone_depth(all_coords[perim_roof_idxs]) * BASE_ADIABATIC_FRAC
    )
    old_roof_core_depth = old_ground_core_depth = old_roof_perim_depth * core_2_perim
    core_origin = min(all_coords[core_roof_idxs, :, 1].flatten())

    # move perim adiabatic line for roof
    pr = all_coords[perim_roof_idxs]
    pr[pr == old_roof_perim_depth] = perim_roof_depth
    all_coords[perim_roof_idxs] = pr
    # move perim adiabatic line for floor
    pf = all_coords[perim_ground_idxs]
    pf[pf == old_ground_perim_depth] = perim_ground_depth
    all_coords[perim_ground_idxs] = pf
    # move core adiabatic line for roof
    cr = all_coords[core_roof_idxs]
    cr[cr == old_roof_core_depth + core_origin] = core_roof_depth + core_origin
    all_coords[core_roof_idxs] = cr
    # move core adiabatic line for floor
    cf = all_coords[core_ground_idxs]
    cf[cf == old_ground_core_depth + core_origin] = core_ground_depth + core_origin
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


def scale_1D(all_coords, scale, axis):
    min_dim = min(all_coords[:, :, axis].flatten())
    move_1D(all_coords, -min_dim, axis)
    for coords in all_coords:
        max_dim = max(coords[:, axis].flatten())
        coords[:, axis] *= scale
    move_1D(all_coords, min_dim, axis)
    return all_coords


def move_1D(coords, distance, axis):
    try:
        coords[:, :, axis] = coords[:, :, axis] + distance
    except:
        coords[:, axis] = coords[:, axis] + distance
    return coords


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
    for name, window in epjson["FenestrationSurface:Detailed"].items():
        coords_all = get_window_coords(window)
        # Check current wwr
        max_h = max(coords_all[:, 2])
        min_h = min(coords_all[:, 2])
        curr_window_h = max_h - min_h
        all_coords = get_all_coords(epjson["BuildingSurface:Detailed"])
        curr_h = max(all_coords[:, :, 2].flatten())
        logger.info(
            f"Current window to wall ratio is {curr_window_h/curr_h} with height {curr_window_h}"
        )
        # Scale window to needed height - assuming window spans entire width of wall
        new_window_h = wwr_frac * curr_h
        logger.info(f"New window height will be {new_window_h}")
        change = (new_window_h - curr_window_h) / 2  # negative if new window is smaller
        # move max down
        coords_all[:, 2][coords_all[:, 2] == max_h] = max_h + change
        # move min up
        coords_all[:, 2][coords_all[:, 2] == min_h] = min_h - change
        # replace coordinates
        window = replace_window_coords(window, coords_all)
    return epjson


def build_shading(sb_json, angles, radius=10, override=True):
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
                {f"vertex_{j}_coordinate": c[k] for k, j in enumerate(["x", "y", "z"])}
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
