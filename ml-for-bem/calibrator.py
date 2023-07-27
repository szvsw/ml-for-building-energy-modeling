import torch
import torch.nn as nn

from schema import Schema
from surrogate import Surrogate


class Calibrator(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        n_timeseries_input_channels: int,
        timeseries_input_period: int,
        timeseries_input_length: int,
        n_static_input_parameters: int,
        weather_timeseries: torch.Tensor,
        surrogate: Surrogate
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_timeseries_input_channels = n_timeseries_input_channels
        self.timeseries_input_period = timeseries_input_period
        self.timeseries_input_length = timeseries_input_length
        self.n_static_input_parameters = n_static_input_parameters

        # Fixed shared weather vector
        self.weather_timeseries = weather_timeseries

        # TODO: Use initializer constraints from feature engineering
        self._timeseries = nn.Parameter(
            torch.randn(
                ensemble_size,
                self.n_timeseries_input_channels,
                self.timeseries_input_period,
            ),
        )





        width = 4
        height = 3
        # Todo: build automatic parameter builder for geometric parameters etc, some are specified as "known" others as "free" others yet as "constrained", e.g.
        self.height = surrogate.schema["height"].normalize(height) # assume 3m
        self.width = surrogate.schema["width"].normalize(width) # assume 3m

        # Should these be separate for the four directions? identical?  fixed?
        
        self._geo_ratios = nn.Parameter(
            torch.randn(self.ensemble_size, 4)
        ) # all four sides have the same adiabatic roof/ground ratios and core/perim split and depth
        self._wwr = nn.Parameter(
            torch.randn(4*ensemble_size,1)
        ) # unique for the four sides

        # Create four orientations
        orientations = torch.arange(4).reshape(-1, 1).tile((self.ensemble_size, 1)).to(device)
        self.orientations = nn.functional.one_hot(orientations).squeeze()

        self._statics = nn.Parameter(
            torch.randn(ensemble_size, self.n_static_input_parameters),
        )
        self.surrogate = surrogate

    @property
    def timeseries(self):
        return torch.sigmoid(self._timeseries)
    
    @property
    def geo_ratios(self):
        return torch.sigmoid(self._geo_ratios)

    @property
    def wwr(self):
        return torch.sigmoid(self._wwr)

    @property
    def statics(self):
        return torch.sigmoid(self._statics)

    def forward(self):
        # ---------------------
        # insert any constant parameters which are not allowed to change, e.g. geometry.

        # ---------------------
        # repeat once per orientation
        widths = self.width * torch.ones(4*self.ensemble_size,1).to(device)
        heights = self.height * torch.ones(4*self.ensemble_size,1).to(device)
        
        _f2f = self.surrogate.schema["facade_2_footprint"].unnormalize(self.geo_ratios[:,0:1])
        _p2f = self.surrogate.schema["perim_2_footprint"].unnormalize(self.geo_ratios[:,1:2])
        _height = surrogate.schema["height"].unnormalize(self.height) 
        _width = surrogate.schema["width"].unnormalize(self.width) 
        _depths = _height / _f2f
        _areas = _width * _depths
        _perims = _areas * _p2f
        _cores = _areas * (1-_p2f)
        _perims = _areas * _p2f
        areas_norm = (_areas - self.surrogate.area_min) / ( self.surrogate.area_max - self.surrogate.area_min)
        perim_areas_norm = (_perims - self.surrogate.area_perim_min) / ( self.surrogate.area_perim_max - self.surrogate.area_perim_min)
        core_areas_norm = (_cores - self.surrogate.area_core_min) / ( self.surrogate.area_core_max - self.surrogate.area_core_min)

        areas_norm = areas_norm.repeat_interleave(4, axis=0)
        perim_areas_norm = perim_areas_norm.repeat_interleave(4, axis=0)
        core_areas_norm = core_areas_norm.repeat_interleave(4, axis=0)

        # depths = geo_ratios[]
        geo_ratios = self.geo_ratios.repeat_interleave(4,axis=0) # constrain to 0-1
        statics = self.statics.repeat_interleave(4, axis=0)
        statics[:,0] = schema["HeatingSetpoint"].normalize(19)
        statics[:,1] = schema["CoolingSetpoint"].normalize(23)
        statics[:,12] = (0.5 - 0.1) / (0.99 - 0.05) # force shgc to be 0.5
        statics[:,13] = (0.7 - 0.1) / (0.99 - 0.05) # force shgc to be 0.5


        # <insert orientations and wwrs>
        building_vector = torch.concatenate((widths,heights, geo_ratios, self.wwr, self.orientations, statics, areas_norm, perim_areas_norm, core_areas_norm), axis=1)

        # <update any orientation responsive parameters>

        # dummy vec

        # -----------------
        # expand building vector by adding final dimension and tiling
        building_vector = building_vector.reshape(*building_vector.shape,1)
        building_vector = building_vector.repeat_interleave(self.surrogate.output_resolution, axis=2)

        # ---------------------
        # concatenate climate vectors and timeseries
        # (c, t) -> (1,c,t), c = weather file channels count, t=8760
        weather = self.weather_timeseries.reshape(-1, *self.weather_timeseries.shape)

        # Repeat weather per ensemble member per orientation -> (4*ensemble_sice, c, t)
        weather = torch.repeat_interleave(weather, 4 * self.ensemble_size, axis=0)

        # tile a week into a year long schedule for each candidate
        timeseries = self.timeseries.tile(
            (
                1,
                1,
                int(self.timeseries_input_length // self.timeseries_input_period + 1),
            )
        )

        # Trim to 8760
        timeseries = timeseries[:, :, : self.timeseries_input_length]

        # repeat schedules per orientation
        timeseries = timeseries.repeat_interleave(4, axis=0)

        # concatenate climate  and bldg timeseries
        timeseries_vector = torch.concatenate((weather, timeseries), axis=1)

        # Project timeseries to Latent vector
        latent_vect = self.surrogate.timeseries_net(timeseries_vector)

        # concatenate
        x = torch.concatenate((latent_vect, building_vector), axis=1)
        result = self.surrogate.energy_net(x)

        # weight factors

        # ---------------------
        return result, result.sum()


if __name__ == "__main__":
    device='cuda'
    import json
    with open("./data/city_map.json","r") as f:
        city_map = json.load(f)
    schema = Schema()
    # TODO: load in checkpoint and climate vector
    surrogate = Surrogate(schema=schema, checkpoint="batch_permute_lower_lr/batch_permute_lower_lr_202307250612_002_350000.pt")



    empty_batch = schema.generate_empty_storage_batch(n=2)
    schema.update_storage_batch(storage_batch=empty_batch,parameter="base_epw", value=city_map["NY, New York"]["idx"])
    ts = surrogate.get_batch_climate_timeseries(empty_batch)
    print(surrogate.building_params_per_vector, surrogate.latent_size)
    cal = Calibrator(
        ensemble_size=10,
        n_timeseries_input_channels=3,
        timeseries_input_period=24 * 7,
        timeseries_input_length=8760,
        n_static_input_parameters=14,
        weather_timeseries=torch.Tensor(ts[0]).to(device),
        surrogate=surrogate
    ).to(device)

    print(sum([param.numel() for param in cal.parameters()]))
    surrogate.energy_net.eval()
    surrogate.timeseries_net.eval()
    for param in surrogate.energy_net.parameters():
        param.requires_grad = False
    for param in surrogate.timeseries_net.parameters():
        param.requires_grad = False



    optim = torch.optim.Adam(cal.parameters(), lr=0.1)

    res = None
    for i in range(1000):
        optim.zero_grad()
        res, sum = cal()
        sum.backward()
        optim.step()
        print(sum.item())
        if sum.item() < 0.1:
            break
    

    for i in range(10):
        print("-"*10 + str(i) + "-"*10)
        ratios = cal.geo_ratios[i].detach().cpu().numpy()
        f2f = ratios[0]
        p2f = ratios[1]
        r2f = ratios[2]
        f2g = ratios[3]

        wwr = cal.wwr.detach().cpu().numpy()[i*4:i*4+4].flatten()

        statics = cal.statics.detach().cpu().numpy()
        heatset = statics[i, 0]
        coolset = statics[i, 1]
        lpd = statics[i, 2]
        epd = statics[i, 3]
        pd = statics[i, 4]
        inf = statics[i, 5]
        facademass = statics[i, 6]
        roofmass = statics[i, 7]
        facader = statics[i, 8]
        roofr = statics[i, 9]
        slabr = statics[i, 10]
        winu = statics[i, 11]
        
        print("f2f",schema["facade_2_footprint"].unnormalize(f2f))
        print("p2f",schema["perim_2_footprint"].unnormalize(p2f))
        print("r2f",schema["roof_2_footprint"].unnormalize(r2f))
        print("f2g",schema["footprint_2_ground"].unnormalize(f2g))
        print("wwr",schema["wwr"].unnormalize(wwr))
        print("hsp", schema["HeatingSetpoint"].unnormalize(heatset))
        print("csp", schema["CoolingSetpoint"].unnormalize(coolset))
        print("lpd", schema["LightingPowerDensity"].unnormalize(lpd))
        print("epd", schema["EquipmentPowerDensity"].unnormalize(epd))
        print("pd", schema["PeopleDensity"].unnormalize(pd))
        print("inf", schema["Infiltration"].unnormalize(inf))
        print("facademass", schema["FacadeMass"].unnormalize(facademass))
        print("roofmass", schema["RoofMass"].unnormalize(roofmass))
        print("facader", schema["FacadeRValue"].unnormalize(facader))
        print("roofr", schema["RoofRValue"].unnormalize(roofr))
        print("slabr", schema["SlabRValue"].unnormalize(slabr))
        print("winu", winu*(schema["WindowSettings"].max[0] -schema["WindowSettings"].min[0])+schema["WindowSettings"].min[0])

        print("\n")
        print(res[i].detach().cpu().numpy())
        if i == 9:
            scheds = cal.timeseries[i].detach().cpu().numpy()
            print(scheds.shape)
        print("\n")


