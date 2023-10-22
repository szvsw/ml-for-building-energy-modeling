from typing import List, Optional, Union
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


class Conv1DDepthwiseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "same",
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        intermediate_dim = 1 * in_channels
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=intermediate_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            ),
            nn.Conv1d(
                in_channels=intermediate_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(out_channels),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "same",
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1DStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i, kernel_size in enumerate(kernel_sizes):
            layers.append(
                Conv1DBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                )
            )
        self.layers = nn.Sequential(*layers)

        self.skip_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if out_channels != in_channels
            else nn.Identity(),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = self.skip_layer(x)
        x_out = self.layers(x)
        x_out = x_out + x_skip

        return nn.functional.relu(x_out)


class Conv1DStageConfig:
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int]):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={in_channels}"
        s += ", out_channels={out_channels}"
        s += ", kernel_sizes={kernel_sizes}"
        s += ")"
        return s.format(**self.__dict__)

    @classmethod
    def Base(cls, in_channels):
        return [
            cls(in_channels, 64, [49, 25, 9]),
            cls(64, 128, [49, 25, 9]),
            cls(128, 128, [49, 25, 9]),
        ]


class ConvNet(nn.Module):
    def __init__(
        self,
        stage_configs: List[Conv1DStageConfig],
        latent_channels: int,
        latent_length: int,
    ):
        super().__init__()
        self.stage_configs = stage_configs
        stages: List[nn.Module] = []
        for stage_conf in self.stage_configs:
            stages.append(
                Conv1DStage(
                    in_channels=stage_conf.in_channels,
                    out_channels=stage_conf.out_channels,
                    kernel_sizes=stage_conf.kernel_sizes,
                )
            )
        self.stages = nn.Sequential(*stages)
        pooling = nn.AdaptiveAvgPool1d(latent_length)

        # Step 1: average pooling down to latent length
        # Step 2: Learnable channel size change which is similar to original setup
        # self.final = nn.Sequential(
        #     pooling,
        #     Conv1DBlock(
        #         in_channels=self.stage_configs[-1].out_channels,
        #         out_channels=latent_channels,
        #         kernel_size=latent_length,
        #         stride=1,
        #         padding='same'
        #     )
        # )

        # Step 1: average pooling to latent length
        # Step 2: Learnable channel size change depthwise weighted average
        # Should be equivalent to the one below
        # self.final = nn.Sequential(
        #     pooling,
        #     Permute([0,2,1]),
        #     nn.Linear(
        #         in_features=self.stage_configs[-1].out_channels,
        #         out_features=latent_channels,
        #     ),
        #     Permute([0,2,1]),
        #     nn.BatchNorm1d(latent_channels),
        #     nn.ReLU(),
        # )

        # Step 1: average pooling to latent length
        # Step 2: Learnable channel size change depthwise weighted average
        # self.final = nn.Sequential(
        #     pooling,
        #     Conv1DBlock(
        #         in_channels=self.stage_configs[-1].out_channels,
        #         out_channels=latent_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding='same'
        #     )
        # )

        # Combines learnable downsampling and channel size change, expensive.
        # self.final = Conv1DBlock(
        #     in_channels=self.stage_configs[-1].out_channels,
        #     out_channels=latent_channels,
        #     kernel_size=int(8760/latent_length),
        #     stride=int(8760/latent_length),
        #     padding=0
        # )

        # Step 1: learnable channel size change which is a depthwise weighted average
        # Step 2: learnable downsampling
        # Should be identical to the next example
        # self.final = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.stage_configs[-1].out_channels,
        #         out_channels=latent_channels,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ),
        #     Conv1DBlock(
        #         in_channels=latent_channels,
        #         out_channels=latent_channels,
        #         kernel_size=int(8760 / latent_length),
        #         stride=int(8760 / latent_length),
        #         padding=0,
        #     ),
        # )

        # Step 1: learnable channel size change which is a depthwise weighted average
        # Step 2: learnable downsampling
        self.final = nn.Sequential(
            Permute([0, 2, 1]),
            nn.Linear(
                in_features=self.stage_configs[-1].out_channels,
                out_features=latent_channels,
            ),
            Permute([0, 2, 1]),
            Conv1DBlock(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=int(8760 / latent_length),
                stride=int(8760 / latent_length),
                padding=0,
            ),
        )

        # Step 1: Adaptive pool to "days" length
        # Step 2: learnable downsampling combined with channel size change
        # self.final = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(360),
        #     Conv1DBlock(
        #         in_channels=self.stage_configs[-1].out_channels,
        #         out_channels=latent_channels,
        #         kernel_size=30,
        #         stride=30,
        #         padding=0,
        #     )
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stages(x)
        out = self.final(features)
        return out


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


class EnergyCNN2(nn.Module):
    def __init__(self, in_channels=30, n_feature_maps=128, n_layers=3, out_channels=4):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv1DStage(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_sizes=[1] * n_layers,
                activation=nn.LeakyReLU,
            ),
            Conv1DStage(
                in_channels=n_feature_maps,
                out_channels=out_channels,
                kernel_sizes=[1],
                activation=nn.LeakyReLU,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


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
