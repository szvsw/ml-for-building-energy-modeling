import math
import numpy as np
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def scale_shoebox(
    sb,
    width,
    height,
    floor_2_facade,
    core_2_perim,
    roof_2_footprint,
    ground_2_footprint,
):
    pass


def set_adiabatic_surfaces(
    sb, height, floor_2_facade, core_2_perim, roof_2_footprint, ground_2_footprint
):
    pass


def update_wwr(sb, wwr_frac):
    pass


def vertices_to_coords(vertices):
    coords = []
    for v in vertices:
        coords.append(list(v.values()))
    return coords


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
