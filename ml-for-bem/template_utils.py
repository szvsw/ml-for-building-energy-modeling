# Create a minimum building template generated from
# base Boston building template library
# TODO: Cladding?

import os
from pathlib import Path
import numpy as np
import re
import pandas as pd

# Import window schema
from nrel_uitls import WINDTYPES

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from eppy.bunch_subclass import EpBunch

    import archetypal as ar
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


class minimumTemplate(UmiTemplateLibrary):
    pass

    # def setup_seed(
    #     self,
    #     template_path=os.path.join(
    #         os.getcwd(),
    #         "ml-for-bem",
    #         "data",
    #         "template_libs",
    #         "ConstructionsLibrary.json",
    #     ),
    # ):
    #     self.seed_template = UmiTemplateLibrary.open(template_path)

    def initialize(self, name):  # csv_path, name
        self.Name = name

        # unchanging objects
        self.WindowConstructions = self.make_window_constructions()
        print(self.GlazingMaterials)
        self.OpaqueConstructions = self.make_opaque_constructions()

        self.setup_default_scheds()

        window_settings = []
        for i, window in enumerate(self.WindowConstructions):
            window_settings.append(
                WindowSetting(
                    Name=f"window_setting_{i}",
                    Construction=window,
                    AfnWindowAvailability=self.always_off,
                    ShadingSystemAvailabilitySchedule=self.always_off,
                    ZoneMixingAvailabilitySchedule=self.always_off,
                )
            )
            # List of WindowSetting objects (needed for Umi template creation)
        self.WindowSettings = window_settings

        dhw_setting = DomesticHotWaterSetting(
            Name="dwh_setting",
            IsOn=True,
            WaterSchedule=self.always_on,
            FlowRatePerFloorArea=0.03,
            WaterSupplyTemperature=65,
            WaterTemperatureInlet=10,
        )
        # List of DomesticHotWaterSetting objects (needed for Umi template creation)
        self.DomesticHotWaterSettings = [dhw_setting]

    def setup_default_scheds(self):
        # Always on
        sch_d_on = DaySchedule.from_values(
            Name="AlwaysOn", Values=[1] * 24, Type="Fraction", Category="Day"
        )
        # Always off
        sch_d_off = DaySchedule.from_values(
            Name="AlwaysOff", Values=[0] * 24, Type="Fraction", Category="Day"
        )
        self.DaySchedules = [sch_d_on, sch_d_off]

        # Week schedules
        # Always on
        sch_w_on = WeekSchedule(
            Days=[sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on],
            Category="Week",
            Type="Fraction",
            Name="AlwaysOn",
        )
        # Always off
        sch_w_off = WeekSchedule(
            Days=[
                sch_d_off,
                sch_d_off,
                sch_d_off,
                sch_d_off,
                sch_d_off,
                sch_d_off,
                sch_d_off,
            ],
            Category="Week",
            Type="Fraction",
            Name="AlwaysOff",
        )

        self.WeekSchedules = [sch_w_on, sch_w_off]

        # Year schedules
        # Always on
        dict_on = {
            "$id": 1,
            "Category": "Year",
            "Parts": [
                {
                    "FromDay": 1,
                    "FromMonth": 1,
                    "ToDay": 31,
                    "ToMonth": 12,
                    "Schedule": sch_w_on.to_ref(),
                }
            ],
            "Type": "Fraction",
            "Name": "AlwaysOn",
        }
        self.always_on = YearSchedule.from_dict(
            dict_on, {a.id: a for a in self.WeekSchedules}
        )
        # Always off
        dict_off = {
            "$id": 2,
            "Category": "Year",
            "Parts": [
                {
                    "FromDay": 1,
                    "FromMonth": 1,
                    "ToDay": 31,
                    "ToMonth": 12,
                    "Schedule": sch_w_off.to_ref(),
                }
            ],
            "Type": "Fraction",
            "Name": "AlwaysOff",
        }
        self.always_off = YearSchedule.from_dict(
            dict_off, {a.id: a for a in self.WeekSchedules}
        )

        self.YearSchedules = [self.always_on, self.always_off]

    def make_window_constructions(self):
        airLayer = GasLayer(self.GasMaterials[0], Thickness=0.006)
        glassClear = MaterialLayer(self.GlazingMaterials[0], Thickness=0.003)
        glassLowE = MaterialLayer(self.GlazingMaterials[2], Thickness=0.003)

        singleClear = WindowConstruction(
            Name="single_clr",
            Layers=[glassClear],
            Category="Single",
        )
        dblClear = WindowConstruction(
            Name="dbl_clr", Layers=[glassClear, airLayer, glassClear], Category="Double"
        )
        dblLowE = WindowConstruction(
            Name="dbl_LoE", Layers=[glassLowE, airLayer, glassClear], Category="Double"
        )
        tripleClear = WindowConstruction(
            Name="triple_clr",
            Layers=[glassClear, airLayer, glassClear, airLayer, glassClear],
            Category="Triple",
        )
        tripleLowE = WindowConstruction(
            Name="triple_LoE",
            Layers=[glassLowE, airLayer, glassClear, airLayer, glassClear],
            Category="Triple",
        )
        return [singleClear, dblClear, dblLowE, tripleClear, tripleLowE]

    def make_opaque_constructions(self):
        self.concrete = self.OpaqueMaterials[3]  # Concrete dense
        # woodstud = template.OpaqueMaterials[12] # Wood stud
        self.insulation = self.OpaqueMaterials[11]  # Fiberglass batt
        plywood = self.OpaqueMaterials[10]  # Fiberglass batt
        gypsum = self.OpaqueMaterials[5]  # 14:Gypsum_Board
        cladding = self.OpaqueMaterials[16]  # Vinyl_cladding
        ceramicTile = self.OpaqueMaterials[8]  # 17:Ceramic_Tile

        # concreteLayer = MaterialLayer(concrete, Thickness=0.2)
        # woodstudLayer = MaterialLayer(woodstud, Thickness=0.2)
        # insulationLayer = MaterialLayer(insulation, Thickness=0.2)
        self.claddingLayer = MaterialLayer(cladding, Thickness=0.003)
        self.plywoodLayer = MaterialLayer(plywood, Thickness=0.015)
        self.gypsumLayer = MaterialLayer(gypsum, Thickness=0.015)

        # Structure definition TODO
        # StructureInformation using OpaqueMaterial objects
        mass_ratio = MassRatio(
            Name="mass_ratio_default",
            Material=plywood,
            HighLoadRatio=1,
            NormalRatio=1,
        )
        struct_definition = StructureInformation(
            Name="struct_info_default", MassRatios=[mass_ratio]
        )
        # List of StructureInformation objects (needed for Umi template creation)
        self.StructureDefinitions = [struct_definition]
        self.StructureInformations = [struct_definition]

        self.default_floor = OpaqueConstruction(
            Name="floor",
            Layers=[
                MaterialLayer(self.insulation, Thickness=0.2),
                MaterialLayer(self.concrete, Thickness=0.2),
                MaterialLayer(ceramicTile, Thickness=0.02),
            ],
            Surface_Type="Ground",
            Outside_Boundary_Condition="Zone",
        )

        self.default_partition = OpaqueConstruction(
            Name="partition",
            Layers=[
                self.gypsumLayer,
                MaterialLayer(self.insulation, Thickness=0.02),
                self.gypsumLayer,
            ],
            Surface_Type="Partition",
            Outside_Boundary_Condition="Zone",
        )
        return [
            self.default_floor,
            self.default_partition,
        ]

    def loMass_facade(self, r_val):
        t = self.insulation.Conductivity * r_val
        print(t)
        loMass_facade = OpaqueConstruction(
            Name="loMass_facade",
            Layers=[
                self.claddingLayer,
                self.plywoodLayer,
                MaterialLayer(self.insulation, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Facade",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(loMass_facade)
        return loMass_facade

    def hiMass_facade(self, r_val):
        t = self.insulation.Conductivity * r_val
        hiMass_facade = OpaqueConstruction(
            Name="hiMass_facade",
            Layers=[
                MaterialLayer(self.concrete, Thickness=0.2),
                self.plywoodLayer,
                MaterialLayer(self.insulation, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Facade",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(hiMass_facade)
        return hiMass_facade

    def loMass_roof(self, r_val):
        t = self.insulation.Conductivity * r_val
        woodShingle = self.OpaqueMaterials[14]  # 23:Slate_Tile
        loMass_roof = OpaqueConstruction(
            Name="loMass_roof",
            Layers=[
                MaterialLayer(woodShingle, Thickness=0.02),
                self.plywoodLayer,
                MaterialLayer(self.insulation, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Roof",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(loMass_roof)
        return loMass_roof

    def hiMass_roof(self, r_val):
        t = self.insulation.Conductivity * r_val
        slateTile = self.OpaqueMaterials[14]  # 23:Slate_Tile
        hiMass_roof = OpaqueConstruction(
            Name="hiMass_roof",
            Layers=[
                MaterialLayer(slateTile, Thickness=0.02),
                self.plywoodLayer,
                MaterialLayer(self.insulation, Thickness=t),
                MaterialLayer(self.concrete, Thickness=0.15),
                self.gypsumLayer,
            ],
            Surface_Type="Roof",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(hiMass_roof)
        return hiMass_roof

    def vent_settings(self, ach):
        # TODO
        achstring = f"{str(round(ach, 2)).replace('.', 'p')}ach"
        # check if already exists
        for i, x in enumerate(self.VentilationSettings):
            if achstring in x.Name:
                return self.VentilationSettings[i]
        else:
            # VentilationSetting using YearSchedule objects
            vent_setting = VentilationSetting(
                Name=f"vent_setting_{achstring}",
                Infiltration=ach,
                NatVentSchedule=self.always_off,
                ScheduledVentilationSchedule=self.always_off,
            )
            # List of VentilationSetting objects (needed for Umi template creation)
            self.VentilationSettings.append(vent_setting)
            return vent_setting

    def conditioning_settings(self, cop_h, cop_c=3.0):
        copstring = f"{round(cop_h*100)}cop"

        for i, x in enumerate(self.ZoneConditionings):
            if copstring in x.Name:
                return self.ZoneConditionings[i]
        else:
            # ZoneConditioning using YearSchedule objects
            zone_conditioning = ZoneConditioning(
                Name=f"zone_conditioning_{copstring}",
                CoolingCoeffOfPerf=cop_c,
                HeatingCoeffOfPerf=cop_h,
                CoolingSchedule=self.always_on,
                HeatingSchedule=self.always_on,
                MechVentSchedule=self.always_off,
            )
            self.ZoneConditionings.append(zone_conditioning)
            return zone_conditioning

    def internal_loads_settings(self, epd, lpd):
        loadsstr = f"{str(round(epd, 2)).replace('.', 'p')}epd_{str(round(lpd, 2)).replace('.', 'p')}lpd"
        # check if already exists
        for i, x in enumerate(self.ZoneLoads):
            if loadsstr in x.Name:
                return self.ZoneLoads[i]
        else:
            # ZoneLoad using YearSchedule objects
            zone_load = ZoneLoad(
                Name=f"zone_load_{loadsstr}",
                EquipmentPowerDensity=epd,
                LightingPowerDensity=lpd,
                EquipmentAvailabilitySchedule=self.always_on,
                LightsAvailabilitySchedule=self.always_on,
                OccupancySchedule=self.always_on,
                PeopleDensity=0.2,
            )
            # List of ZoneLoad objects (needed for Umi template creation)
            self.ZoneLoads.append(zone_load)
            return zone_load

    def define_zone_constructions(self, idx, facade, roof):
        # Perimeter zone
        zone_constr_set_perim = ZoneConstructionSet(
            Name=f"constr_set_perim_{idx}",
            Slab=self.default_floor,
            Roof=roof,
            Partition=self.default_partition,
            Ground=self.default_floor,
            Facade=facade,
        )
        # Core zone
        zone_constr_set_core = ZoneConstructionSet(
            Name=f"constr_set_core_{idx}",
            Slab=self.default_floor,
            Roof=roof,
            Partition=self.default_partition,
            IsPartitionAdiabatic=True,
            Ground=self.default_floor,
            Facade=facade,
        )
        self.ZoneConstructionSets.extend([zone_constr_set_perim, zone_constr_set_core])
        return zone_constr_set_perim, zone_constr_set_core

    def clean_name(self, type, vintage):
        type = re.sub("[\W_]+", "", type.lower())
        vintage = re.sub("[\W_]+", "", vintage.lower())
        return type + "_" + vintage

    def construct_building_template(
        self,
        bld_template_name,
        idx,
        ach,
        cop_h,
        epd,
        lpd,
        r_floor,
        r_wall,
        r_roof,
        window,
        wwr=0.4,
        cop_c=3.0,
    ):
        ZoneLoads = self.internal_loads_settings(epd=epd, lpd=lpd)
        print(self.ZoneLoads)

        ZoneConditioning = self.conditioning_settings(cop_h=cop_h, cop_c=cop_c)
        print(self.ZoneConditionings)

        ZoneVentilation = self.vent_settings(ach=ach)
        print(self.VentilationSettings)

        wind_idx = WINDTYPES[window]

        # make high mass and low mass version

        # TODO change r values
        perimHi_cons, coreHi_cons = self.define_zone_constructions(
            idx, self.hiMass_facade(r_wall), self.hiMass_roof(r_roof)
        )
        perimLo_cons, coreLo_cons = self.define_zone_constructions(
            idx, self.loMass_facade(r_wall), self.loMass_roof(r_roof)
        )

        print("RVALUE FACADE", perimLo_cons.Facade.r_value)
        print("RVALUE ROOF", perimLo_cons.Roof.r_value)

        # Perimeter zone
        perimLo = ZoneDefinition(
            Name=f"Perim_zone_LoMass_{idx}",
            Constructions=perimLo_cons,  #
            Loads=ZoneLoads,
            Conditioning=ZoneConditioning,
            Ventilation=ZoneVentilation,
            DomesticHotWater=self.DomesticHotWaterSettings[0],
            DaylightMeshResolution=1,
            DaylightWorkplaneHeight=0.8,
            InternalMassConstruction=self.default_partition,
            InternalMassExposedPerFloorArea=1.05,
            WindowConstruction=self.WindowConstructions[wind_idx],
            Windows=self.WindowSettings[wind_idx],
        )
        perimHi = ZoneDefinition(
            Name=f"Perim_zone_HiMass_{idx}",
            Constructions=perimHi_cons,  #
            Loads=ZoneLoads,
            Conditioning=ZoneConditioning,
            Ventilation=ZoneVentilation,
            DomesticHotWater=self.DomesticHotWaterSettings[0],
            DaylightMeshResolution=1,
            DaylightWorkplaneHeight=0.8,
            InternalMassConstruction=self.default_partition,
            InternalMassExposedPerFloorArea=1.05,
            WindowConstruction=self.WindowConstructions[wind_idx],
            Windows=self.WindowSettings[wind_idx],
        )
        coreLo = ZoneDefinition(
            Name=f"Core_zone_LoMass_{idx}",
            Constructions=coreLo_cons,  #
            Loads=ZoneLoads,
            Conditioning=ZoneConditioning,
            Ventilation=ZoneVentilation,
            DomesticHotWater=self.DomesticHotWaterSettings[0],
            DaylightMeshResolution=1,
            DaylightWorkplaneHeight=0.8,
            InternalMassConstruction=self.default_partition,
            InternalMassExposedPerFloorArea=1.05,
            WindowConstruction=self.WindowConstructions[wind_idx],
            Windows=self.WindowSettings[wind_idx],
        )
        coreHi = ZoneDefinition(
            Name=f"Core_zone_HiMass_{idx}",
            Constructions=coreHi_cons,  #
            Loads=ZoneLoads,
            Conditioning=ZoneConditioning,
            Ventilation=ZoneVentilation,
            DomesticHotWater=self.DomesticHotWaterSettings[0],
            DaylightMeshResolution=1,
            DaylightWorkplaneHeight=0.8,
            InternalMassConstruction=self.default_partition,
            InternalMassExposedPerFloorArea=1.05,
            WindowConstruction=self.WindowConstructions[wind_idx],
            Windows=self.WindowSettings[wind_idx],
        )
        # add to List of Zone objects (needed for Umi template creation)
        self.ZoneDefinitions.extend([perimLo, coreLo, perimHi, coreHi])

        # BuildingTemplate using Zone, StructureInformation and WindowSetting objects
        building_template_loMass = BuildingTemplate(
            Name=f"{bld_template_name}_LoMass_{idx}",
            Core=coreLo,
            Perimeter=perimLo,
            Structure=self.StructureDefinitions[0],
            Windows=self.WindowSettings[wind_idx],
            DefaultWindowToWallRatio=wwr,
        )

        building_template_hiMass = BuildingTemplate(
            Name=f"{bld_template_name}_HiMass_{idx}",
            Core=coreHi,
            Perimeter=perimHi,
            Structure=self.StructureDefinitions[0],
            Windows=self.WindowSettings[wind_idx],
            DefaultWindowToWallRatio=wwr,
        )

        self.BuildingTemplates.extend(
            [building_template_loMass, building_template_hiMass]
        )

    def construct_cz_templates(self, templates_df, name):
        self.initialize(name)

        for idx, row in templates_df.iterrows():
            template_name = self.clean_name(
                row["Dependency=Geometry Building Type RECS"],
                row["Dependency=Vintage UBEM"],
            )
            print(idx, " out of ", templates_df.shape[0])
            print("Building building template for ", template_name)
            self.construct_building_template(
                bld_template_name=template_name,
                idx=idx,
                ach=row["ACH"],
                cop_h=row["COP_h"],
                epd=row["EPD"],
                lpd=row["LPD"],
                r_floor=row["R_floor"],
                r_wall=row["R_wall"],
                r_roof=row["R_roof"],
                window=row["Windows"],
                wwr=row["WWR"] / 100,
            )
        self.save(
            path_or_buf=os.path.join(
                os.getcwd(),
                "ml-for-bem",
                "data",
                "template_libs",
                "cz_libs",
                name + ".json",
            )
        )

        return self

    def test_idf(self):
        """
        Test new templates in EnergyPlus
        """


if __name__ == "__main__":
     template_path = os.path.join(
        os.getcwd(), "ml-for-bem", "data", "template_libs", "ConstructionsLibrary.json"
    )
    buildings_df_path = "C:/Users/zoele/Dropbox (MIT)/Downgrades/UBEM_res_templates"
    cz_templatelist = os.listdir(buildings_df_path)
    cz_templatelist = [x for x in cz_templatelist if "residentialtemplates" in x]
    for csv_name in cz_templatelist:
        cz_df = pd.read_csv(os.path.join(buildings_df_path, csv_name), index_col=0)

        seed_template = UmiTemplateLibrary.open(template_path)

        template = minimumTemplate(
            OpaqueMaterials=seed_template.OpaqueMaterials,
            GasMaterials=seed_template.GasMaterials,
            GlazingMaterials=seed_template.GlazingMaterials,
        )

        template.construct_cz_templates(cz_df, csv_name.split(".")[0])