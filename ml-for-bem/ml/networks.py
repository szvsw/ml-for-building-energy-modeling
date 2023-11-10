from typing import List, Optional, Union, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        nn.init.normal_(self.conv.weight, mean=0, std=np.sqrt(1. / self.conv.in_channels))
        
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

    @classmethod
    def Small(cls, in_channels):
        return [
            cls(in_channels, 16, [25, 16, 9]),
            cls(16, 32, [25, 16, 9]),
            cls(32, 64, [25, 16, 9]),
        ]
    
    @classmethod
    def Mini(cls, in_channels):
        return [
            cls(in_channels, 16, [16, 9, 4]),
            cls(16, 16, [16, 9, 4]),
            cls(16, 32, [16, 9, 4]),
        ]
    
    @classmethod
    def MiniFunnel(cls, in_channels):
        return [
            cls(in_channels, 128, [16, 9, 4]),
            cls(16, 16, [16, 9, 4]),
            cls(16, 128, [16, 9, 4]),
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
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(360),
            Conv1DBlock(
                in_channels=self.stage_configs[-1].out_channels,
                out_channels=latent_channels,
                kernel_size=30,
                stride=30,
                padding=0,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stages(x)
        out = self.final(features)
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
            config = [hidden_dim]*block_depth
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
