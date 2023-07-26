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
        aspect_ratio: float,
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
        self.aspect_ratio = aspect_ratio

        # TODO: Use initializer constraints from feature engineering
        self.timeseries = nn.Parameter(
            torch.rand(
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
        
        self.geo_ratios = nn.Parameter(
            torch.rand(self.ensemble_size, 4)
        ) # all four sides have the same adiabatic roof/ground ratios and core/perim split and depth
        self.wwr = nn.Parameter(
            torch.rand(4*ensemble_size,1)
        ) # unique for the four sides

        # Create four orientations
        orientations = torch.arange(4).reshape(-1, 1).tile((self.ensemble_size, 1)).to(device)
        self.orientations = nn.functional.one_hot(orientations).squeeze()

        self.statics = nn.Parameter(
            torch.rand(ensemble_size, self.n_static_input_parameters),
        )
        self.surrogate = surrogate

    def forward(self):
        # ---------------------
        # insert any constant parameters which are not allowed to change, e.g. geometry.

        # ---------------------
        # repeat once per orientation
        widths = self.width * torch.ones(4*self.ensemble_size,1).to(device)
        heights = self.height * torch.ones(4*self.ensemble_size,1).to(device)
        geo_ratios = self.geo_ratios.repeat_interleave(4,axis=0)
        statics = self.statics.repeat_interleave(4, axis=0)

        # <insert orientations and wwrs>
        building_vector = torch.concatenate((widths,heights, geo_ratios, self.wwr, self.orientations,statics), axis=1)

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
        return result.sum()


if __name__ == "__main__":
    device='cuda'
    schema = Schema()
    surrogate = Surrogate(schema=schema)
    cal = Calibrator(
        ensemble_size=20,
        n_timeseries_input_channels=3,
        timeseries_input_period=24 * 7,
        timeseries_input_length=8760,
        n_static_input_parameters=17,
        weather_timeseries=torch.rand(8, 8760).to(device),
        aspect_ratio=2,
        surrogate=surrogate
    ).to(device)

    print(sum([param.numel() for param in cal.parameters()]))
    surrogate.energy_net.eval()
    surrogate.timeseries_net.eval()
    for param in surrogate.energy_net.parameters():
        param.requires_grad = False
    for param in surrogate.timeseries_net.parameters():
        param.requires_grad = False

    optim = torch.optim.Adam(cal.parameters(), lr=0.001)

    for i in range(1000):
        optim.zero_grad()
        res = cal()
        res.backward()
        optim.step()
        print(res)
        

