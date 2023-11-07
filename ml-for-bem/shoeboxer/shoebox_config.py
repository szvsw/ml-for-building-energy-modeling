class ShoeboxConfiguration:
    """
    Stateful class for shoebox object args
    """

    __slots__ = (
        "width",
        "height",
        "perim_depth",
        "core_depth",
        "adiabatic_partition_flag",
        # "floor_2_facade",
        # "core_2_perim",
        "roof_2_footprint",
        "ground_2_footprint",
        "wwr",
        "orientation",
        "shading_vect",
    )

    def __init__(self):
        """
        Builder throws error if core is less than 2.
        Set adiabatic partition flag. Check that this works - adiabatic wall & various core depths.
        """
        pass

    @classmethod
    def from_umi(cls):
        pass  # TODO


# TODO
# Add bounds and checks to params
