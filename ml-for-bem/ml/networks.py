from typing import List, Optional, Type, Union, Literal, Tuple

import numpy as np
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


class Conv1DDepthThenPointBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        expansion_factor: int = 1,
        padding: Union[int, str] = "same",
        activation: nn.Module = nn.LeakyReLU,
        intermediate_activation: bool = False,
        norm_layer: Literal["batch", "layer"] = "batch",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.expansion_factor = expansion_factor
        intermediate_dim = expansion_factor * in_channels
        layers = []
        if kernel_size != None:
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=intermediate_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            )
            # Initialize the weights
            nn.init.normal_(conv.weight, mean=0, std=np.sqrt(1.0 / conv.in_channels))

            # If using biases, initialize them to zero
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

            self.depthwise_conv = conv
            layers.append(conv)
            if intermediate_activation:
                layers.append(activation())
        else:
            intermediate_dim = in_channels
            # if intermediate_activation:
            #     raise ValueError(
            #         "intermediate_activation must be False if kernel_size == None"
            #     )
        self.intermediate_dim = intermediate_dim
        self.out_channels = out_channels
        conv = nn.Conv1d(
            in_channels=intermediate_dim,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Initialize the weights
        nn.init.normal_(conv.weight, mean=0, std=np.sqrt(1.0 / conv.in_channels))

        # If using biases, initialize them to zero
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        self.pointwise_conv = conv
        layers.append(conv)
        if norm_layer == "batch":
            layers.append(nn.BatchNorm1d(out_channels))
        elif norm_layer == "layer":
            layers.append(nn.LayerNorm(normalized_shape=[out_channels]))
        elif norm_layer == None:
            pass
        else:  # no normalization
            raise ValueError(
                f"norm_layer must be None, 'batch' or 'layer', got {norm_layer}"
            )
        layers.append(activation())

        self.block = nn.Sequential(
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv1DDepthThanPointStage(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: List[int],
        kernel_sizes: List[int],
        expansion_factors: List[int],
        activation: nn.Module = nn.LeakyReLU,
        intermediate_activation: bool = False,
        skip_normalization: bool = False,
        norm_layer: Literal["batch", "layer"] = "batch",
        pooling_size: int = None,
    ):
        super().__init__()
        skip_norm = nn.Identity()
        if skip_normalization:
            if norm_layer == "batch":
                skip_norm = nn.BatchNorm1d(out_dims[-1])
            elif norm_layer == "layer":
                skip_norm = nn.LayerNorm(normalized_shape=[out_dims[-1]])
            else:
                raise ValueError("skip_normalization requires norm_layer to be set")
        self.skip_transform = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dims[-1],
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if in_dim != out_dims[-1]
            else nn.Identity(),
            skip_norm,
        )

        blocks = []

        if kernel_sizes is None:
            kernel_sizes = [None] * len(out_dims)
        for out_dim, kernel_size, expansion_factor in zip(
            out_dims, kernel_sizes, expansion_factors
        ):
            blocks.append(
                Conv1DDepthThenPointBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    expansion_factor=expansion_factor,
                    activation=activation,
                    intermediate_activation=intermediate_activation,
                    norm_layer=norm_layer,
                )
            )
            in_dim = out_dim
        self.blocks = nn.Sequential(*blocks)
        self.pooling = nn.AdaptiveAvgPool1d(pooling_size) if pooling_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = self.skip_transform(x)
        x_out = self.blocks(x)
        x = x_out + x_skip
        if self.pooling:
            x = self.pooling(x)
        return nn.functional.leaky_relu(x)


class Conv1DNextNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        expansion_factor: int = 1,
        activation: nn.Module = nn.LeakyReLU,
        intermediate_activation: bool = False,
        skip_normalization: bool = False,
        norm_layer: Literal["batch", "layer"] = "batch",
        flatten: bool = False,
        final_shape: Tuple[int, int] = (4, 12),
        stages=[
            {
                "out_dims": [256, 128, 64],
                "kernel_sizes": [5, 5, 5],
                "pooling_size": None,
            },
            {
                "out_dims": [64, 64, 64],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [64, 64, 64],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [64, 64, 64],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [64, 64, 64],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [64, 64, 64],
                "kernel_sizes": None,
                "pooling_size": 365,
            },
            {
                "out_dims": [128, 256, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "pooling_size": [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
            {
                "out_dims": [512, 512, 512],
                "kernel_sizes": None,
                "pooling_size": None,
            },
        ],
    ):
        super().__init__()
        self.flatten = flatten
        self.activation = activation
        self.intermediate_activation = intermediate_activation
        self.norm_layer = norm_layer
        self.skip_normalization = skip_normalization
        self.final_shape = final_shape

        net = []
        stage_in_dim = in_dim
        for stage in stages:
            if isinstance(stage["pooling_size"], list):
                timestep_sizes = stage["pooling_size"]
                net.append(CustomPooling(timestep_sizes))
            else:
                net.append(
                    Conv1DDepthThanPointStage(
                        in_dim=stage_in_dim,
                        **stage,
                        expansion_factors=[expansion_factor] * len(stage["out_dims"]),
                        activation=activation,
                        intermediate_activation=intermediate_activation,
                        skip_normalization=skip_normalization,
                        norm_layer=norm_layer,
                    )
                )
                stage_in_dim = stage["out_dims"][-1]
        if self.flatten:
            net.append(nn.Flatten())
            final_pooling_size = stages[-1]["pooling_size"]
            final_in_dim = stage_in_dim * final_pooling_size
            final_out_dim = int(np.product(self.final_shape))
            net.append(nn.Linear(final_in_dim, final_out_dim))
        else:
            net.append(
                Conv1DDepthThenPointBlock(
                    in_channels=stage_in_dim,
                    out_channels=self.final_shape[0],
                    kernel_size=None,
                    activation=activation,
                    norm_layer=None,
                    stride=1,
                    expansion_factor=1,
                    padding=0,
                )
            )
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.flatten:
            x = x.reshape(x.shape[0], *self.final_shape)
        return x


class CustomPooling(nn.Module):
    def __init__(self, timestep_sizes):
        super().__init__()
        self.timestep_sizes = timestep_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is (batch_size, features, timesteps)

        we will pick out the first timestep_size[0] values, then the next timestep_size[1] values, etc.
        and average them
        """

        timestep_groups = []
        start = 0
        for timestep_size in self.timestep_sizes:
            timestep_groups.append(
                (x[:, :, start : start + timestep_size]).mean(dim=-1)
            )
            start += timestep_size
        # combine the averages so that we have (batch_size, features, len(timestep_sizes))
        return torch.stack(timestep_groups, dim=-1)


# class Conv1DDepthwiseBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int = 1,
#         padding: Union[int, str] = "same",
#         activation: nn.Module = nn.LeakyReLU,
#     ):
#         super().__init__()
#         intermediate_dim = 1 * in_channels
#         self.block = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=in_channels,
#                 out_channels=intermediate_dim,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 groups=in_channels,
#             ),
#             nn.Conv1d(
#                 in_channels=intermediate_dim,
#                 out_channels=out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#             ),
#             nn.BatchNorm1d(out_channels),
#             activation(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = "same",
        activation: nn.Module = nn.SELU,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch = nn.BatchNorm1d(out_channels)
        self.act = activation()

        # Initialize the weights
        nn.init.normal_(
            self.conv.weight, mean=0, std=np.sqrt(1.0 / self.conv.in_channels)
        )

        # If using biases, initialize them to zero
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.batch(self.conv(x)))


class Conv1DBlockWithoutInit(nn.Module):
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
        activation=nn.SELU,
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

        return nn.functional.leaky_relu(x_out)


class Conv1DStageConfig:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
        out_length: int = -1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.out_length = out_length

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={in_channels}"
        s += ", out_channels={out_channels}"
        s += ", kernel_sizes={kernel_sizes}"
        s += ", out_length={out_length}"
        s += ")"
        return s.format(**self.__dict__)

    @classmethod
    def Large(cls, in_channels):
        return [
            cls(in_channels, 128, [49, 25, 9], 1024),
            cls(128, 128, [49, 25, 9], 512),
            cls(128, 128, [49, 25, 9], 128),
            cls(128, 128, [49, 25, 9], 12),
        ]

    @classmethod
    def Base(cls, in_channels):
        return [
            cls(in_channels, 128, [25, 16, 9], 1024),
            cls(128, 128, [25, 16, 9], 128),
            cls(128, 128, [25, 16, 9], 12),
        ]

    @classmethod
    def Medium(cls, in_channels):
        return [
            cls(in_channels, 128, [25, 16, 9], 1024),
            cls(128, 128, [9, 9, 9], 128),
            cls(128, 128, [9, 9, 9], 12),
        ]

    @classmethod
    def Meso(cls, in_channels):
        return [
            cls(in_channels, 128, [8, 5, 3], 1024),
            cls(128, 32, [8, 5, 3], 128),
            cls(32, 32, [8, 5, 3], 12),
        ]

    @classmethod
    def Small(cls, in_channels):
        return [
            cls(in_channels, 16, [25, 16, 9], 1024),
            cls(16, 32, [25, 16, 9], 128),
            cls(32, 64, [25, 16, 9], 12),
        ]

    @classmethod
    def Mini(cls, in_channels):
        return [
            cls(in_channels, 16, [16, 9, 4], 1024),
            cls(16, 16, [16, 9, 4], 128),
            cls(16, 32, [16, 9, 4], 12),
        ]

    @classmethod
    def MiniOdd(cls, in_channels):
        return [
            cls(in_channels, 16, [9, 5, 3], 1024),
            cls(16, 16, [9, 5, 3], 1024),
            cls(16, 32, [9, 5, 3], 1024),
            # cls(32, 32, [9, 5, 3], 128),
            # cls(32, 32, [9, 5, 3], 128),
        ]

    @classmethod
    def VeryMini(cls, in_channels):
        return [
            cls(in_channels, 16, [3, 3, 3], 1024),
            cls(16, 16, [3, 3, 3], 1024),
            cls(16, 32, [3, 3, 3], 1024),
            # cls(32, 32, [9, 5, 3], 128),
            # cls(32, 32, [9, 5, 3], 128),
        ]

    @classmethod
    def MiniFunnel(cls, in_channels):
        return [
            cls(in_channels, 128, [16, 9, 4]),
            cls(16, 16, [16, 9, 4]),
            cls(16, 128, [16, 9, 4]),
        ]


class ConvNet2(nn.Module):
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
            stages.append(nn.AdaptiveAvgPool1d(stage_conf.out_length))
        self.stages = nn.Sequential(*stages)
        self.final = nn.Sequential(
            Conv1DBlock(
                in_channels=self.stage_configs[-1].out_channels,
                out_channels=latent_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )


class ConvNet(nn.Module):
    def __init__(
        self,
        stage_configs: List[Conv1DStageConfig],
        latent_channels: int,
        latent_length: int,
        custom_pooling: bool = False,
        big_skip_enabled: bool = False,
    ):
        super().__init__()
        self.big_skip_enabled = big_skip_enabled
        self.custom_pooling = custom_pooling
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
        # self.final = nn.Sequential(
        #     Permute([0, 2, 1]),
        #     nn.Linear(
        #         in_features=self.stage_configs[-1].out_channels,
        #         out_features=latent_channels,
        #     ),
        #     Permute([0, 2, 1]),
        #     Conv1DBlock(
        #         in_channels=latent_channels,
        #         out_channels=latent_channels,
        #         kernel_size=int(8760 / latent_length),
        #         stride=int(8760 / latent_length),
        #         padding=0,
        #     ),
        # )

        # Step 1: Adaptive pool to "days" length
        # TODO: experiment with going to 12hrs rather than 24 hrs, i.e. AdaptiveAvgPool1D(720)
        # Step 2: learnable downsampling combined with channel size change
        self.final = (
            nn.Sequential(
                nn.AdaptiveAvgPool1d(360),
                Conv1DBlock(
                    in_channels=self.stage_configs[-1].out_channels,
                    out_channels=latent_channels,
                    kernel_size=30,
                    stride=30,
                    padding=0,
                ),
            )
            if not self.custom_pooling
            else nn.Sequential(
                nn.AvgPool1d(24),
                Conv1DBlock(
                    in_channels=self.stage_configs[-1].out_channels,
                    out_channels=latent_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                CustomPooling([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
            )
        )
        self.big_skip = (
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.stage_configs[0].in_channels,
                    out_channels=self.stage_configs[-1].out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.BatchNorm1d(self.stage_configs[-1].out_channels),
            )
            if self.big_skip_enabled
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stages(x)
        out = self.final(
            features + self.big_skip(x) if self.big_skip_enabled else features
        )
        return out


class EnergyCNN2(nn.Module):
    def __init__(
        self, in_channels=30, n_feature_maps=128, n_layers=3, n_blocks=4, out_channels=4
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            Conv1DStage(
                in_channels=in_channels,
                out_channels=n_feature_maps,
                kernel_sizes=[1] * n_layers,
                activation=nn.SELU,
            ),
            *[
                Conv1DStage(
                    in_channels=n_feature_maps,
                    out_channels=n_feature_maps,
                    kernel_sizes=[1] * n_layers,
                    activation=nn.SELU,
                )
                for _ in range(n_blocks - 2)
            ],
            Conv1DStage(
                in_channels=n_feature_maps,
                out_channels=out_channels,
                kernel_sizes=[1],
                activation=nn.Identity,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class SkipBlock(nn.Module):
    def __init__(
        self,
        dims: List[int] = [20, 30, 40],
        act: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dims = dims
        self.dropout = dropout
        layers = [nn.Dropout(dropout)]
        for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:]):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    act(),
                )
            )
        self.layers = nn.Sequential(*layers)
        self.skip = (
            nn.Linear(self.dims[0], self.dims[-1])
            if self.dims[0] != self.dims[-1]
            else nn.Identity()
        )

    def forward(self, x) -> torch.Tensor:
        return self.layers(x) + self.skip(x)


class MLP(nn.Module):
    """
    A PyTorch Neural Network with a basic MLP Regressor Architecture.
    It includes several fully connected layers, along with skip connections for odd layers.

    """

    def __init__(
        self,
        input_dim=50,
        hidden_dim: int = 100,
        block_depth: int = 3,
        block_count: int = 4,
        output_dim=6,
        activation: Type[nn.Module] = nn.SiLU,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Width of hidden layers within each skip block
            block_depth (int): Number of hidden layers within each skip block
            block_count (int): Number of skip blocks in network
            output_dim (int): Width of output
            activation (function (in: torch.Tensor) -> torch.Tensor): Activation function
        """

        super().__init__()
        self.input_dim = input_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
        )

        hidden_blocks: List[nn.Module] = []
        for i in range(block_count):
            config = [hidden_dim] * block_depth
            if i < block_count - 2:
                config = config + [
                    hidden_dim
                ]  # append the output dimension of the next layer
            hidden_blocks.append(
                SkipBlock(
                    dims=config,
                    act=activation,
                    dropout=dropout,
                )
            )
        self.hidden_blocks = nn.Sequential(*hidden_blocks)
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_blocks(x)
        x = self.final_layer(x)
        return x
