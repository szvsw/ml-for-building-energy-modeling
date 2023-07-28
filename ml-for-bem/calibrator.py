import torch
import torch.nn as nn

from schema import Schema
from surrogate import Surrogate


class ShoeboxCalibrator(nn.Module):
    def __init__(
        self,
        ensemble_size: int,
        n_timeseries_input_channels: int,
        timeseries_input_period: int,
        timeseries_input_length: int,
        n_static_input_parameters: int,
        surrogate: Surrogate,
        width: float=4,
        height: float=3,
        depth: float=10,
        orientation: int = 0,
        epw_idx: int = 0,
        roof_2_footprint: float = 0,
        footprint_2_ground: float = 0,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_timeseries_input_channels = n_timeseries_input_channels
        self.timeseries_input_period = timeseries_input_period
        self.timeseries_input_length = timeseries_input_length
        self.n_static_input_parameters = n_static_input_parameters

        # Fixed shared weather vector
        empty_batch = schema.generate_empty_storage_batch(n=2)
        schema.update_storage_batch(storage_batch=empty_batch,parameter="base_epw", value=epw_idx)
        schema.update_storage_batch(storage_batch=empty_batch,parameter="orientation", value=orientation)
        ts = surrogate.get_batch_climate_timeseries(empty_batch)
        self.weather_timeseries = torch.Tensor(ts[0]).to(device)

        # TODO: Use initializer constraints from feature engineering
        self._timeseries = nn.Parameter(
            torch.randn(
                ensemble_size,
                self.n_timeseries_input_channels,
                self.timeseries_input_period,
            ),
        )

        self._perim_2_footprint = nn.Parameter(
            torch.randn(self.ensemble_size, 1)
        ) 

        self._wwr = nn.Parameter(
            torch.randn(ensemble_size,1)
        ) 

        self._statics = nn.Parameter(
            torch.randn(ensemble_size, self.n_static_input_parameters),
        )




        self.__width = width
        self.__height = height
        self.__depth = depth
        self.__facade_2_footprint = self.__width / self.__depth
        self.__roof_2_footprint = roof_2_footprint
        self.__footprint_2_ground = footprint_2_ground

        self.__orientation = orientation
        self.orientation = nn.functional.one_hot(torch.arange(4))[self.__orientation].to(device)
        # TODO: build automatic parameter builder for geometric parameters etc, some are specified as "known" others as "free" others yet as "constrained", e.g.

        # Create four orientations
        # orientations = torch.arange(4).reshape(-1, 1).tile((self.ensemble_size, 1)).to(device)
        # self.orientations = nn.functional.one_hot(orientations).squeeze()

        self.surrogate = surrogate

    @property
    def width(self):
        return self.surrogate.schema["width"].normalize(self.__width)

    @property
    def height(self):
        return self.surrogate.schema["height"].normalize(self.__height)

    @property
    def facade_2_footprint(self):
        return self.surrogate.schema["facade_2_footprint"].normalize(self.__facade_2_footprint)

    @property
    def roof_2_footprint(self):
        return self.surrogate.schema["roof_2_footprint"].normalize(self.__roof_2_footprint)

    @property
    def footprint_2_ground(self):
        return self.surrogate.schema["footprint_2_ground"].normalize(self.__footprint_2_ground)
    
    @property
    def __area(self):
        return self.__depth * self.__width
    
    @property
    def area(self):
        return (self.__area - self.surrogate.area_min) / (self.surrogate.area_max - self.surrogate.area_min)

    @property
    def __perim_area(self):
        return self.__area * self.surrogate.schema["perim_2_footprint"].unnormalize(self.perim_2_footprint)

    @property
    def perim_area(self):
        return (self.__perim_area - self.surrogate.area_perim_min) / (self.surrogate.area_perim_max - self.surrogate.area_perim_min)

    @property
    def __core_area(self):
        return self.__area * (1-self.surrogate.schema["perim_2_footprint"].unnormalize(self.perim_2_footprint))

    @property
    def core_area(self):
        return (self.__core_area - self.surrogate.area_core_min) / (self.surrogate.area_core_max - self.surrogate.area_core_min)


    @property
    def timeseries(self):
        return torch.sigmoid(self._timeseries)
    
    @property
    def perim_2_footprint(self):
        return torch.sigmoid(self._perim_2_footprint)

    @property
    def wwr(self):
        return torch.sigmoid(self._wwr)

    @property
    def statics(self):
        return torch.sigmoid(self._statics)

    def forward(self):
        widths = torch.Tensor((self.roof_2_footprint,)).repeat(self.ensemble_size).reshape(-1,1).to(device)
        heights = torch.Tensor((self.roof_2_footprint,)).repeat(self.ensemble_size).reshape(-1,1).to(device)
        f2f = torch.Tensor((self.facade_2_footprint,)).repeat(self.ensemble_size).reshape(-1,1).to(device)
        # --->perim
        r2f = torch.Tensor((self.roof_2_footprint,)).repeat(self.ensemble_size).reshape(-1,1).to(device)
        f2g = torch.Tensor((self.footprint_2_ground,)).repeat(self.ensemble_size).reshape(-1,1).to(device)
        # --> orient
        orientations = self.orientation.reshape(1,4).repeat((self.ensemble_size,1)).to(device)
        # --> statics
        areas = torch.Tensor(self.area.repeat(self.ensemble_size).reshape(-1,1)).to(device)
        perim_areas = torch.Tensor(self.perim_area).to(device)
        core_areas = torch.Tensor(self.core_area).to(device)


        # ---------------------
        # insert any constant parameters which are not allowed to change, e.g. geometry.
        self.statics[:,12] = (0.5 - 0.1) / (0.99 - 0.05) # force shgc to be 0.5
        self.statics[:,13] = (0.7 - 0.1) / (0.99 - 0.05) # force vlt to be 0.7


        # <insert orientations and wwrs>
        building_vector = torch.concatenate((widths,heights, f2f,self.perim_2_footprint, r2f, f2g, self.wwr, orientations, self.statics, areas, perim_areas, core_areas), axis=1)

        # ---------------------
        # concatenate climate vectors and timeseries
        # (c, t) -> (1,c,t), c = weather file channels count, t=8760
        weather = self.weather_timeseries.reshape(-1, *self.weather_timeseries.shape)

        # Repeat weather per ensemble member
        weather = torch.repeat_interleave(weather, self.ensemble_size, axis=0)

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

        # concatenate climate  and bldg timeseries
        timeseries_vector = torch.concatenate((weather, timeseries), axis=1)

        # Project timeseries to Latent vector
        latent_vect = self.surrogate.timeseries_net(timeseries_vector)
        building_vector = building_vector.reshape(*building_vector.shape,1).repeat((1,1,self.surrogate.output_resolution))

        # concatenate
        x = torch.concatenate((latent_vect, building_vector), axis=1)
        result = self.surrogate.energy_net(x)

        # weight factors

        # ---------------------
        return result


if __name__ == "__main__":
    device='cuda'
    import json
    with open("./data/city_map.json","r") as f:
        city_map = json.load(f)
    schema = Schema()
    # TODO: load in checkpoint and climate vector
    surrogate = Surrogate(schema=schema, checkpoint="batch_permute_lower_lr/batch_permute_lower_lr_202307250612_002_350000.pt")

    result_idx = 500000
    target = surrogate.results["eui_normalized"][result_idx]
    vector = surrogate.full_storage_batch[result_idx]
    orientation = schema["orientation"].extract_storage_values(vector).astype(int)
    epw_idx = schema["base_epw"].extract_storage_values(vector).astype(int)
    width = schema["width"].extract_storage_values(vector)
    height = schema["height"].extract_storage_values(vector)
    f2f = schema["facade_2_footprint"].extract_storage_values(vector)
    r2f = schema["roof_2_footprint"].extract_storage_values(vector)
    f2g = schema["footprint_2_ground"].extract_storage_values(vector)
    depth = height/f2f



    cal = ShoeboxCalibrator(
        ensemble_size=25,
        n_timeseries_input_channels=3,
        timeseries_input_period=24 * 14,
        timeseries_input_length=8760,
        n_static_input_parameters=14,
        surrogate=surrogate,
        width=width,
        height=height,
        depth=depth,
        roof_2_footprint=r2f,
        footprint_2_ground=f2g,
        orientation=orientation,
        epw_idx=epw_idx,
    ).to(device)

    print(sum([param.numel() for param in cal.parameters()]))
    surrogate.energy_net.eval()
    surrogate.timeseries_net.eval()
    for param in surrogate.energy_net.parameters():
        param.requires_grad = False
    for param in surrogate.timeseries_net.parameters():
        param.requires_grad = False



    optim = torch.optim.Adam(cal.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    target = torch.Tensor(target).unsqueeze(0).tile(cal.ensemble_size,1,1).to(device)

    res = None
    for i in range(500):
        optim.zero_grad()
        result = cal()
        loss = loss_fn(target,result)
        loss.backward()
        optim.step()
        print(i, loss.item())
    

    result = cal()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,4)
    for i in range(cal.ensemble_size):
        for j in range(4):
            tar = result[i,j]
            if j == 0:
                tar = tar*(surrogate.eui_perim_heating_max - surrogate.eui_perim_heating_min) + surrogate.eui_perim_heating_min
            elif j == 1:
                tar = tar*(surrogate.eui_perim_cooling_max - surrogate.eui_perim_cooling_min) + surrogate.eui_perim_cooling_min
            elif j == 2:
                tar = tar*(surrogate.eui_core_heating_max - surrogate.eui_core_heating_min) + surrogate.eui_core_heating_min
            elif j == 3:
                tar = tar*(surrogate.eui_core_cooling_max - surrogate.eui_core_cooling_min) + surrogate.eui_core_cooling_min
            axs[j].plot(tar.detach().cpu().numpy(),"o", markersize=1, lw=0.6)
        print("-"*10 + str(i) + "-"*10)
        f2f = cal.facade_2_footprint
        r2f = cal.roof_2_footprint
        f2g = cal.footprint_2_ground

        p2f = cal.perim_2_footprint.detach().cpu().numpy()
        wwr = cal.wwr[i].detach().cpu().numpy()[i*4:i*4+4].flatten()

        statics = cal.statics[i].detach().cpu().numpy()
        heatset = statics[0]
        coolset = statics[1]
        lpd = statics[2]
        epd = statics[3]
        pd = statics[4]
        inf = statics[5]
        facademass = statics[6]
        roofmass = statics[7]
        facader = statics[8]
        roofr = statics[9]
        slabr = statics[10]
        winu = statics[11]
        
        # print("f2f",schema["facade_2_footprint"].unnormalize(f2f))
        print("p2f",schema["perim_2_footprint"].unnormalize(p2f))
        # print("r2f",schema["roof_2_footprint"].unnormalize(r2f))
        # print("f2g",schema["footprint_2_ground"].unnormalize(f2g))
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
        if i == 0:
            scheds = cal.timeseries[i].detach().cpu().numpy()
        print("\n")
    for i in range(4):
        tar = target[0,i]
        if i == 0:
            tar = tar*(surrogate.eui_perim_heating_max - surrogate.eui_perim_heating_min) + surrogate.eui_perim_heating_min
        elif i == 1:
            tar = tar*(surrogate.eui_perim_cooling_max - surrogate.eui_perim_cooling_min) + surrogate.eui_perim_cooling_min
        elif i == 2:
            tar = tar*(surrogate.eui_core_heating_max - surrogate.eui_core_heating_min) + surrogate.eui_core_heating_min
        elif i == 3:
            tar = tar*(surrogate.eui_core_cooling_max - surrogate.eui_core_cooling_min) + surrogate.eui_core_cooling_min
        axs[i].plot(tar.detach().cpu().numpy(),"o",markersize=1, linestyle="dashed", lw=1.5, alpha=0.5)
    fig.tight_layout()
    plt.show()


