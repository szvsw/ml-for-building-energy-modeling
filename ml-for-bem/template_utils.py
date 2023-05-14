# Create a minimum building template generated from
# base Boston building template library

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
    from pyumi.shoeboxer.shoebox import ShoeBox

from nrel_uitls import RESTYPES


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
            Name="d_AlwaysOn", Values=[1] * 24, Type="Fraction", Category="Day"
        )
        # Always off
        sch_d_off = DaySchedule.from_values(
            Name="d_AlwaysOff", Values=[0] * 24, Type="Fraction", Category="Day"
        )
        self.DaySchedules.extend([sch_d_on, sch_d_off])

        # Week schedules
        # Always on
        sch_w_on = WeekSchedule(
            Days=[sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on, sch_d_on],
            Category="Week",
            Type="Fraction",
            Name="w_AlwaysOn",
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
            Name="w_AlwaysOff",
        )

        self.WeekSchedules.extend([sch_w_on, sch_w_off])

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
            "Name": "y_AlwaysOn",
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
            "Name": "y_AlwaysOff",
        }
        self.always_off = YearSchedule.from_dict(
            dict_off, {a.id: a for a in self.WeekSchedules}
        )
        self.YearSchedules.extend([self.always_on, self.always_off])

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
        self.concrete_facade = self.OpaqueMaterials[3].duplicate()  # Concrete dense
        self.concrete_facade.Name = self.concrete_facade.Name + "_facade"
        self.concrete_roof = self.OpaqueMaterials[3].duplicate()  # Concrete dense
        self.concrete_roof.Name = self.concrete_roof.Name + "_roof"
        self.concrete_ground = self.OpaqueMaterials[3].duplicate()  # Concrete dense
        self.concrete_ground.Name = self.concrete_ground.Name + "_ground"
        self.insulation = self.OpaqueMaterials[11]  # Fiberglass batt
        self.insulation_facade = self.OpaqueMaterials[11].duplicate()  # Fiberglass batt
        self.insulation_facade.Name = self.insulation_facade.Name + "_facade"
        self.insulation_roof = self.OpaqueMaterials[11].duplicate()  # insulation dense
        self.insulation_roof.Name = self.insulation_roof.Name + "_roof"
        self.insulation_ground = self.OpaqueMaterials[
            11
        ].duplicate()  # insulation dense
        self.insulation_ground.Name = self.insulation_ground.Name + "_ground"

        self.OpaqueMaterials.extend(
            [
                self.concrete_facade,
                self.concrete_roof,
                self.concrete_ground,
                self.insulation_facade,
                self.insulation_roof,
                self.insulation_ground,
            ]
        )

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
                MaterialLayer(self.insulation_ground, Thickness=0.2),
                MaterialLayer(self.concrete_ground, Thickness=0.2),
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
        t = self.insulation_facade.Conductivity * r_val
        loMass_facade = OpaqueConstruction(
            Name="loMass_facade",
            Layers=[
                self.claddingLayer,
                self.plywoodLayer,
                MaterialLayer(self.insulation_facade, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Facade",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(loMass_facade)
        return loMass_facade

    def hiMass_facade(self, r_val):
        t = self.insulation_facade.Conductivity * r_val
        hiMass_facade = OpaqueConstruction(
            Name="hiMass_facade",
            Layers=[
                MaterialLayer(self.concrete_facade, Thickness=0.1),
                self.plywoodLayer,
                MaterialLayer(self.insulation_facade, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Facade",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(hiMass_facade)
        return hiMass_facade

    # def loMass_roof(self, r_val):
    #     t = self.insulation.Conductivity * r_val
    #     woodShingle = self.OpaqueMaterials[14]  # 23:Slate_Tile
    #     loMass_roof = OpaqueConstruction(
    #         Name="loMass_roof",
    #         Layers=[
    #             MaterialLayer(woodShingle, Thickness=0.02),
    #             self.plywoodLayer,
    #             MaterialLayer(self.insulation, Thickness=t),
    #             self.gypsumLayer,
    #         ],
    #         Surface_Type="Roof",
    #         Outside_Boundary_Condition="Outdoors",
    #     )
    #     self.OpaqueConstructions.append(loMass_roof)
    #     return loMass_roof

    def hiMass_roof(self, r_val):
        t = self.insulation_roof.Conductivity * r_val
        # slateTile = self.OpaqueMaterials[14]  # 23:Slate_Tile
        hiMass_roof = OpaqueConstruction(
            Name="hiMass_roof",
            Layers=[
                # MaterialLayer(slateTile, Thickness=0.02),
                MaterialLayer(self.concrete_roof, Thickness=0.1),
                self.plywoodLayer,
                MaterialLayer(self.insulation_roof, Thickness=t),
                self.gypsumLayer,
            ],
            Surface_Type="Roof",
            Outside_Boundary_Condition="Outdoors",
        )
        self.OpaqueConstructions.append(hiMass_roof)
        return hiMass_roof

    def vent_settings(self, ach):
        achstring = f"{str(round(ach, 2)).replace('.', 'p')}ach"
        # check if already exists
        for i, x in enumerate(self.VentilationSettings):
            if achstring in x.Name:
                return self.VentilationSettings[i]
        else:
            # VentilationSetting using YearSchedule objects
            vent_setting = VentilationSetting(
                Name=f"vent_setting_{achstring}",
                IsInfiltrationOn=True,
                Infiltration=ach,
                IsScheduledVentilationOn=False,
                ScheduledVentilationSchedule=self.always_off,
                IsNatVentOn=False,
                NatVentSchedule=self.always_off,
            )
            # List of VentilationSetting objects (needed for Umi template creation)
            self.VentilationSettings.append(vent_setting)
            return vent_setting

    def conditioning_settings(self, cop_h, cop_c=3.0):
        cop_h = 1  # OVERRIDE
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
                IsHeatingOn=True,
                HeatingSchedule=self.always_on,
                IsCoolingOn=True,
                CoolingSchedule=self.always_on,
                IsMechVentOn=True,
                MechVentSchedule=self.always_on,
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
                IsEquipmentOn=True,
                IsLightingOn=True,
                IsPeopleOn=True,
                EquipmentPowerDensity=epd,
                LightingPowerDensity=lpd,
                EquipmentAvailabilitySchedule=self.YearSchedules[2],
                LightsAvailabilitySchedule=self.YearSchedules[1],
                OccupancySchedule=self.YearSchedules[0],
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
        # print(self.ZoneLoads)

        ZoneConditioning = self.conditioning_settings(cop_h=cop_h, cop_c=cop_c)
        # print(self.ZoneConditionings)

        ZoneVentilation = self.vent_settings(ach=ach)
        # print(self.VentilationSettings)

        wind_idx = WINDTYPES[window]

        # make high mass and low mass version

        perimHi_cons, coreHi_cons = self.define_zone_constructions(
            idx, self.hiMass_facade(r_wall), self.hiMass_roof(r_roof)
        )
        perimLo_cons, coreLo_cons = self.define_zone_constructions(
            idx, self.loMass_facade(r_wall), self.hiMass_roof(r_roof)
        )

        print("RVALUE FACADE", perimLo_cons.Facade.r_value)
        print("HEAT CAP FACADE", perimLo_cons.Facade.heat_capacity_per_unit_wall_area)
        print("THICKNESS FACADE LAYERS")
        for layer in perimLo_cons.Facade.Layers:
            print(layer)

        print("RVALUE ROOF", perimLo_cons.Roof.r_value)
        print("HEAT CAP ROOF", perimLo_cons.Roof.heat_capacity_per_unit_wall_area)

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
            Name=f"{bld_template_name}_MASS_0",
            Core=coreLo,
            Perimeter=perimLo,
            Structure=self.StructureDefinitions[0],
            Windows=self.WindowSettings[wind_idx],
            DefaultWindowToWallRatio=wwr,
        )

        building_template_hiMass = BuildingTemplate(
            Name=f"{bld_template_name}_MASS_1",
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
            # template_name = self.clean_name(
            #     row["Dependency=Geometry Building Type RECS"],
            #     row["Dependency=Vintage UBEM"],
            # )
            print("\n", "*" * 50, f"\n{idx} out of {templates_df.shape[0]}")
            print(row["Dependency=Vintage UBEM"])
            v_idx = 0
            if row["Dependency=Vintage UBEM"] == "1946-1980":
                v_idx = 1
            elif row["Dependency=Vintage UBEM"] == "1981-2003":
                v_idx = 2
            elif row["Dependency=Vintage UBEM"] == "2004+":
                v_idx = 3

            template_name = f"PROG_{RESTYPES[row['Dependency=Geometry Building Type RECS']]:02d}_VINTAGE_{v_idx:02d}"
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
        print(self.BuildingTemplates)
        self.save(
            path_or_buf=os.path.join(
                os.getcwd(),
                "ml-for-bem",
                "data",
                "template_libs",
                "cz_libs",
                "residential",
                name + ".json",
            )
        )

        return self


def test_template(lib, epw_path, outdir):
    """
    Test new templates in EnergyPlus
    """
    for template in lib.BuildingTemplates:
        wwr_map = {0: 0, 90: 0, 180: 0.4, 270: 0}  # N is 0, E is 90
        # Convert to coords
        width = 3.0
        depth = 10.0
        perim_depth = 3.0
        height = 3.0
        zones_data = [
            {
                "name": "Perim",
                "coordinates": [
                    (width, 0),
                    (width, perim_depth),
                    (0, perim_depth),
                    (0, 0),
                ],
                "height": height,
                "num_stories": 1,
                "zoning": "by_storey",
            },
            {
                "name": "Core",
                "coordinates": [
                    (width, perim_depth),
                    (width, depth),
                    (0, depth),
                    (0, perim_depth),
                ],
                "height": height,
                "num_stories": 1,
                "zoning": "by_storey",
            },
        ]

        sb = ShoeBox.from_template(
            building_template=template,
            zones_data=zones_data,
            wwr_map=wwr_map,
        )

        # Set floor and roof geometry for each zone
        for surface in sb.getsurfaces(surface_type="roof"):
            name = surface.Name
            name = name.replace("Roof", "Ceiling")
            sb.add_adiabatic_to_surface(surface, name, 0.4)
        for surface in sb.getsurfaces(surface_type="floor"):
            name = surface.Name
            name = name.replace("Floor", "Int Floor")
            sb.add_adiabatic_to_surface(surface, name, 0.4)
        # Internal partition and glazing
        # Orientation
        sb.outputs.add_basics().apply()
        out_df = sb.simulate(
            epw=epw_path,
            output_directory=outdir,
            annual=True,
            keep_data_err=True,
            process_files=True,
            verbose=False,
        )
        print(out_df)


if __name__ == "__main__":
    overwrite = True
    sim = False

    template_path = os.path.join(
        os.getcwd(), "ml-for-bem", "data", "template_libs", "ConstructionsLibrary.json"
    )
    buildings_df_path = "C:/Users/zoele/Dropbox (MIT)/Downgrades/UBEM_res_templates"
    cz_templatelist = os.listdir(buildings_df_path)
    cz_templatelist = [x for x in cz_templatelist if "residentialtemplates" in x]
    cz_templatelist = [x for x in cz_templatelist if ".csv" in x]
    for csv_name in cz_templatelist:
        tpath = os.path.join(
            os.getcwd(),
            "ml-for-bem",
            "data",
            "template_libs",
            "cz_libs",
            "residential",
        )
        existing = os.listdir(tpath)
        outdir = os.path.join(
            os.getcwd(),
            "ml-for-bem",
            "data",
            "template_libs",
            "cz_libs",
            "residential",
            "epresults",
        )
        epw_path = os.path.join(
            os.getcwd(),
            "ml-for-bem",
            "data",
            "epws",
            "CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw",
        )

        name = csv_name.split(".")[0].split("_")[1]
        print("#" * 20, "Running tests for ", name, "#" * 20)

        if overwrite:
            cz_df = pd.read_csv(os.path.join(buildings_df_path, csv_name), index_col=0)
            print("\n", "*" * 50, "\nTEMPLATES FOR CZ ", csv_name)

            seed_template = UmiTemplateLibrary.open(template_path)

            template = minimumTemplate(
                OpaqueMaterials=seed_template.OpaqueMaterials,
                GasMaterials=seed_template.GasMaterials,
                GlazingMaterials=seed_template.GlazingMaterials,
                DaySchedules=seed_template.DaySchedules,
                WeekSchedules=seed_template.WeekSchedules,
                YearSchedules=seed_template.YearSchedules,
            )
            template.construct_cz_templates(cz_df, name)

        elif name + ".json" in existing:
            template = UmiTemplateLibrary.open(os.path.join(tpath, name + ".json"))
        if sim:
            test_template(template, epw_path, outdir)
