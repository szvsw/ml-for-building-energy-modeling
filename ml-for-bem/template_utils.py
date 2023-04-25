# Create a minimum building template generated from
# base Boston building template library
# TODO: Cladding?

import os
import numpy as np

from eppy.bunch_subclass import EpBunch

from archetypal import UmiTemplateLibrary
from archetypal import IDF, settings
from archetypal.eplus_interface import EnergyPlusVersion
from archetypal.simple_glazing import calc_simple_glazing
from archetypal.template.building_template import BuildingTemplate
from archetypal.template.conditioning import ZoneConditioning, EconomizerTypes
from archetypal.template.dhw import DomesticHotWaterSetting
from archetypal.template.load import DimmingTypes, ZoneLoad
from archetypal.template.materials.gas_layer import GasLayer
from archetypal.template.materials.gas_material import GasMaterial
from archetypal.template.materials.glazing_material import GlazingMaterial
from archetypal.template.materials.material_layer import MaterialLayer
from archetypal.template.materials.nomass_material import NoMassMaterial
from archetypal.template.materials.opaque_material import OpaqueMaterial
from archetypal.template.constructions.opaque_construction import OpaqueConstruction
from archetypal.template.constructions.base_construction import (
    ConstructionBase,
    LayeredConstruction,
)
from archetypal.template.schedule import (
    DaySchedule,
    UmiSchedule,
    WeekSchedule,
    YearSchedule,
    YearSchedulePart,
)
from archetypal.template.structure import MassRatio, StructureInformation
from archetypal.template.umi_base import UniqueName
from archetypal.template.ventilation import VentilationSetting
from archetypal.template.constructions.window_construction import WindowConstruction
from archetypal.template.window_setting import WindowSetting
from archetypal.template.zone_construction_set import ZoneConstructionSet
from archetypal.template.zonedefinition import ZoneDefinition
from archetypal.template.constructions.internal_mass import InternalMass
from archetypal.utils import reduce

# Import window schema
from nrel_uitls import WINDTYPES


class minimumTemplate(UmiTemplateLibrary):
    pass

    def define_window_constructions(self):
        glazingMaterials = self.GlazingMaterials
        glassClear = self.GlazingMaterials[0]
        glassLowE = self.GlazingMaterials[0]
        # singleClearMetal = WindowConstruction(
        #     Layers = [glassLayer, airLayer, glassLayer]
        # )
        # singleClearNonMetal
        # singleClearMetalStorm
        # singleClearNonMetalStorm
        # dblClearMetalAir
        # dblClearNonMetalAir
        # dblClearMetalAirStorm
        # dblClearNonMetalAirStorm
        # dblLoweNonMetalAir
        # trpLoweNonMetalAir

    def define_wall_constructions(self):
        # woodStud -
        # CMU
        # Brick
        # Concrete
        pass


if __name__ == "__main__":
    template_path = os.path.join(
        os.getcwd(), "ml-for-bem", "data", "template_libs", "BostonTemplateLibrary.json"
    )

    seed_template = UmiTemplateLibrary.open(template_path)

    OpaqueMaterials = seed_template.OpaqueMaterials
    GasMaterials = seed_template.GasMaterials
    GlazingMaterials = seed_template.GlazingMaterials

    template = minimumTemplate(
        OpaqueMaterials=OpaqueMaterials,
        GasMaterials=GasMaterials,
        GlazingMaterials=GlazingMaterials,
    )
    print(template.GlazingMaterials)
