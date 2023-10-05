class ShoeboxConfiguration:
    """
    Stateful class for shoebox object args
    """

    __slots__ = (
        "width",
        "height",
        "floor_2_facade",
        "core_2_perim",
        "roof_2_footprint",
        "ground_2_footprint",
        "wwr",
        "orientation",
        "shading_vect",
    )

    def __init__(self):
        pass

    @classmethod
    def from_umi(cls):
        pass  # TODO


# TODO
# Add bounds and checks to params
