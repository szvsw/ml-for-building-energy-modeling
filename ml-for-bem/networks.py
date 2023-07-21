from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(
        self,
        dims: List[int],
    ):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)

class LayerNorm1D(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,1)
        F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0,2,1)
        return x

class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        """Implements a ConvNeXt block as per https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
        but with some slight adaptations for the 1D context.

        The block expects inputs with shape `[N,C,T]`, where `N` is the batch size, `C` is the channel count, and
        `T` is the number of steps in the TimeSeries.

        Args:
            dim (int): The number C of input and output channels for the block

        """
        super().__init__()

        # use depthwise convulution: each input channel gets a single, unique filter.
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=49,
                padding='same',  # 24
                groups=dim,
                bias=True,
            ),
            Permute([0, 2, 1]),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(
                in_features=dim,
                out_features=4 * dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=4 * dim,
                out_features=dim,
                bias=True,
            ),
            Permute([0, 2, 1]),
        )
        # TODO: implement stochastic depth and layer_scale? https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected = self.block(input) # block
        output = projected + input  # Skip
        return output


class ConvNeXtStage1D(nn.Module):
    """
    A stack of identical ConvNeXt 1D Blocks which together form a stage.  a ConvNext network is typically made up of 4 stages.
    """

    def __init__(
        self,
        n_layers: int = 3,
        input_channels: int = 96,
        output_channels: Optional[int] = None,
    ):
        """Creates a stack of identical ConvNeXt 1D Blocks which together form a stage.

        Args:
            n_layers (int): The number of blocks to stack up in the stage.
            input_channels (int): The number of channels `C` for each block in the stage.
            output_channels (Optional[int]): If not `None`, the final output will change the number of layers
                but downsample the sequence length by a factor of 2D proccess using a strided 1D conv.
        """
        super().__init__()

        modules: List[nn.Module] = []

        # Create the Blocks
        blocks: List[nn.Module] = []
        for i in range(n_layers):
            # TODO: see convnext 2d implementation for stochastic depth prob updates if implemented.
            block = ConvNeXtBlock1D(dim=input_channels)
            blocks.append(block)

        # Optionally change output size.
        modules.append(nn.Sequential(*blocks))

        if output_channels is not None:
            downscaler: List[nn.Module] = []
            downscaler.append(LayerNorm1D(input_channels, eps=1e-6))
            downscaler.append(
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
            modules.append(nn.Sequential(*downscaler))

        self.blocks = nn.Sequential(*modules)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output


class ConvNeXt1DStageConfig:
    """
    Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py

    Stores information listed at Section 3 of the ConvNeXt paper
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = output_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={output_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @classmethod
    def Tiny(cls):
        return [
            cls(96, 192, 3),
            cls(192, 384, 3),
            cls(384, 768, 9),
            cls(768, None, 3),
        ]

    @classmethod
    def Small(cls):
        return [
            cls(96, 192, 3),
            cls(192, 384, 3),
            cls(384, 768, 27),
            cls(768, None, 3),
        ]

    @classmethod
    def Medium(cls):
        return [
            cls(128, 256, 3),
            cls(256, 512, 3),
            cls(512, 1024, 27),
            cls(1024, None, 3),
        ]


    @classmethod
    def Large(cls):
        return [
            cls(192, 384, 3),
            cls(384, 768, 3),
            cls(768, 1536, 27),
            cls(1536, None, 3),
        ]


class ConvNeXt1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        series_output_size: int,
        stage_configs: List[ConvNeXt1DStageConfig],
    ):
        """
        Creates a ConvNeXt network with multiple stages, each of which may have many blocks.

        Args:
            n_timeseries_channels (int): The number of timeseries channels in the dataset.
            n_timesteps_in_output (int): The number of timesteps to reduce the final dimension to using adaptive average pooling
            stage_configs (List[ConvNeXt1DStageConfig]): A list where each element provides configuration for the
                blocks within that stage as well as a final channel expansion/contraction output count.
        """
        super().__init__()

        # TODO: Stem Cell stage for channel matching:
        # start at N x C1 x T
        # end at N x C2 x T / ?
        # C1 = n_timeseries_channels
        # C2 = channel dimension of each block of in the first stage
        first_convblock_in_channels = stage_configs[0].input_channels
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=first_convblock_in_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                groups=1,
                bias=True,
            ),
            LayerNorm1D(
                first_convblock_in_channels,
                eps=1e-6
            ),
        )

        total_stage_blocks = sum(conf.num_layers for conf in stage_configs)
        stages: List[nn.Module] = []
        for conf in stage_configs:
            stage = ConvNeXtStage1D(
                n_layers=conf.num_layers,
                input_channels=conf.input_channels,
                output_channels=conf.out_channels,
            )
            stages.append(stage)
        self.features = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool1d(series_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        # TODO: finish implementing regressor
        return x 


class EnergyTimeseriesCNNBlockA(nn.Module):
    def __init__(
        self,
        in_channels=11,
        n_feature_maps=64,
    ):
        super().__init__()

        self.n_feature_maps = n_feature_maps

        self.input_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_size=49,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.mid_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps,
                out_channels=n_feature_maps,
                kernel_size=25,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.final_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps,
                out_channels=n_feature_maps,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.skip_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

    def forward(self, x):
        x_skip = self.skip_layer(x)

        x_out = self.input_convolutional_layer(x)
        x_out = nn.functional.relu(x_out)

        x_out = self.mid_convolutional_layer(x_out)
        x_out = nn.functional.relu(x_out)

        x_out = self.final_convolutional_layer(x_out)

        x_out = x_out + x_skip

        return nn.functional.relu(x_out)


class EnergyTimeseriesCNNBlockB(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_feature_maps=128,
    ):
        super().__init__()

        self.n_feature_maps = n_feature_maps

        self.input_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_size=49,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.mid_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps,
                out_channels=n_feature_maps,
                kernel_size=25,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.final_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps,
                out_channels=n_feature_maps,
                kernel_size=9,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.skip_layer = nn.BatchNorm1d(n_feature_maps)

    def forward(self, x):
        x_skip = self.skip_layer(x)

        x_out = self.input_convolutional_layer(x)
        x_out = nn.functional.relu(x_out)

        x_out = self.mid_convolutional_layer(x_out)
        x_out = nn.functional.relu(x_out)

        x_out = self.final_convolutional_layer(x_out)

        x_out = x_out + x_skip

        return nn.functional.relu(x_out)


class AnnualEnergyCNN(nn.Module):
    def __init__(
        self,
        out_channels=22,
        n_feature_maps=64,
    ):
        super().__init__()

        self.resblock_1 = EnergyTimeseriesCNNBlockA(n_feature_maps=n_feature_maps)

        self.resblock_2 = EnergyTimeseriesCNNBlockA(
            in_channels=n_feature_maps, n_feature_maps=n_feature_maps * 2
        )

        # no need to expand channels in third layer because they are equal
        self.resblock_3 = EnergyTimeseriesCNNBlockB(
            in_channels=n_feature_maps * 2, n_feature_maps=n_feature_maps * 2
        )

        # FOR ANNUAL
        self.GlobalAveragePool = nn.AvgPool1d(
            kernel_size=8760
        )  # 1D? average across all feature maps
        self.linear = nn.Linear(
            in_features=n_feature_maps * 2, out_features=out_channels
        )

    def forward(self, x):
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.GlobalAveragePool(x)
        x = x.squeeze(-1)
        x = self.linear(x)
        return nn.functional.relu(x)


class MonthlyEnergyCNN(nn.Module):
    def __init__(
        self,
        in_channels=8,
        out_channels=8,
        n_feature_maps=64,
    ):
        super().__init__()

        self.resblock_1 = EnergyTimeseriesCNNBlockA(n_feature_maps=n_feature_maps)

        self.resblock_2 = EnergyTimeseriesCNNBlockA(
            in_channels=n_feature_maps, n_feature_maps=n_feature_maps * 2
        )

        # no need to expand channels in third layer because they are equal
        self.resblock_3 = EnergyTimeseriesCNNBlockB(
            in_channels=n_feature_maps * 2, n_feature_maps=n_feature_maps * 2
        )

        # FOR MONTHLY (out is 2x12)
        self.month_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps * 2,
                out_channels=out_channels,
                kernel_size=30,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.pooling = nn.AvgPool1d(kernel_size=730)

    def forward(self, x):
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.pooling(x)
        x = self.month_convolutional_layer(x)
        return nn.functional.relu(x)


class EnergyCNN(torch.nn.Module):
    def __init__(self, in_channels=30, n_feature_maps=64, out_channels=2):
        super(EnergyCNN, self).__init__()

        # FOR MONTHLY (out is 2x12)
        self.in_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(n_feature_maps),
        )

        self.out_convolutional_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=n_feature_maps,
                out_channels=out_channels,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm1d(out_channels),
        )
        # self.pooling = nn.AvgPool1d(kernel_size=730)

    def forward(self, sample):
        # sample (22+n, 1)
        x = self.in_convolutional_layer(sample)
        x = nn.functional.leaky_relu(x)
        x = self.out_convolutional_layer(x)
        x = nn.functional.relu(x)

        return x
