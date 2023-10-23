from typing import Literal, Union
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ml.networks import EnergyCNN2, ConvNet, Conv1DStageConfig
from ml.data import MinMaxTransform, StdNormalTransform


class MultiModalModel(nn.Module):
    """
    A model that takes in building features, climate data, and schedules and predicts energy
    """

    def __init__(self, timeseries_net: nn.Module, energy_net: nn.Module):
        """
        Args:
            timeseries_net: a network that takes in timeseries data (batch_size, n_channels, n_timesteps) and outputs a latent representation
            energy_net: a network that takes in a latent representation (batch_size, n_latent_channel, n_latent_steps) and outputs energy (batch_size, n_energy_channels, n_energy_steps)
        """
        super().__init__()

        # store the nets
        self.timeseries_net = timeseries_net
        self.energy_net = energy_net

    def forward(
        self,
        building_features: torch.Tensor,
        climates: torch.Tensor,
        schedules: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict energy from building features, climate data, and schedules
        Args:
            building_features: (batch_size, n_features)
            climates: (batch_size, n_climate_channels, n_timesteps)
            schedules: (batch_size, n_schedule_channels, n_timesteps)
        """

        # Combine the climate and schedules along the timeseries channel axis
        timeseries = torch.cat([climates, schedules], dim=1)

        # Pass the timeseries through the timeseries net to get a latent representation
        latent = self.timeseries_net(timeseries)

        # Repeat the building features along the latent timeseries axis
        building_features = building_features.unsqueeze(-1)
        building_features = building_features.repeat(1, 1, latent.shape[-1])

        # Concatenate the building features and latent representation along the channel axis
        input = torch.cat([building_features, latent], dim=1)

        # Pass the concatenated input through the energy net to get energy predictions
        preds = self.energy_net(input)

        return preds


class Surrogate(pl.LightningModule):
    """
    A PyTorch Lightning abstraction for managing a surrogate model which can predict energy from building features, climate data, and schedules
    """

    def __init__(
        self,
        target_transform: Union[MinMaxTransform, StdNormalTransform],
        net_config: Literal["Base", "Small"] = "Small",
        lr: float = 1e-3,
        lr_gamma: float = 0.5,
        latent_factor: int = 4,
        energy_cnn_feature_maps: int = 128,
        energy_cnn_n_layers: int = 3,
        energy_cnn_n_blocks: int = 5,
        timeseries_channels_per_input: int = 10,
        static_features_per_input: int = 10,
        timeseries_channels_per_output: int = 4,
        timeseries_steps_per_output: int = 12,
    ):
        """
        Create the Lighting Module for managing the surrogate

        Args:
            target_transform (nn.Module): a transform to apply to targets.  Should implement the `inverse_transform` method.
            net_config (Literal["Base", "Small"], optional): the configuration of the timeseries net architecture.  Defaults to "Small".
            lr (float, optional): the learning rate.  Defaults to 1e-3.
            lr_gamma (float, optional): the learning rate decay.  Defaults to 0.5. Called after each epoch completes.
            latent_factor (int, optional): The timeseries net will output a latent representation with `latent_factor * static_features_per_input` channels.  Defaults to 4.
            energy_cnn_feature_maps (int, optional): The number of feature maps in the energy net.  Defaults to 128.
            energy_cnn_n_layers (int, optional): The number of layers in each energy net block.  Defaults to 3.
            energy_cnn_n_blocks (int, optional): The number of energy net blocks.  Defaults to 5.
            timeseries_channels_per_input (int, optional): The number of timeseries channels in the input.  Defaults to 10.
            static_features_per_input (int, optional): The number of static features in the input.  Defaults to 10.
            timeseries_channels_per_output (int, optional): The number of timeseries channels in the output.  Defaults to 4.
            timeseries_steps_per_output (int, optional): The number of timesteps in the output.  Defaults to 12.

        Returns:
            Surrogate: the PyTorch Lightning Module which manages the surrogate model

        """

        # TODO: auto configure network dims
        # TODO: hyperparameters for configure block architectures
        # TODO: store weather transform and feature transform, not just target transform
        super().__init__()

        # Store all the hyperparameters
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.net_config = net_config
        self.timeseries_channels_per_input = timeseries_channels_per_input
        self.static_features_per_input = static_features_per_input
        self.latent_factor = latent_factor
        self.latent_channels = self.static_features_per_input * self.latent_factor
        self.energy_cnn_in_size = self.latent_channels + self.static_features_per_input
        self.timeseries_channels_per_output = timeseries_channels_per_output
        self.timeseries_steps_per_output = timeseries_steps_per_output
        self.energy_cnn_feature_maps = energy_cnn_feature_maps
        self.energy_cnn_n_blocks = energy_cnn_n_blocks
        self.energy_cnn_n_layers = energy_cnn_n_layers

        # Create the configuration for the timeseries net
        conf = (
            Conv1DStageConfig.Base(self.timeseries_channels_per_input)
            if self.net_config == "Base"
            else Conv1DStageConfig.Small(self.timeseries_channels_per_input)
        )
        self.conf = conf

        # Save hyperparameters
        self.save_hyperparameters()

        # Create the timeseries net and energy net
        timeseries_net = ConvNet(
            stage_configs=conf,
            latent_channels=self.latent_channels,
            latent_length=self.timeseries_steps_per_output,
        )

        energy_net = EnergyCNN2(
            in_channels=self.energy_cnn_in_size,
            out_channels=self.timeseries_channels_per_output,
            n_feature_maps=self.energy_cnn_feature_maps,
            n_blocks=self.energy_cnn_n_blocks,
            n_layers=self.energy_cnn_n_layers,
        )

        # Create and store the model
        self.model = MultiModalModel(
            timeseries_net=timeseries_net, energy_net=energy_net
        )

        # Store the target transform module
        self.target_transform = target_transform

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=self.lr_gamma,
        )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        return {
            "lr_scheduler_config": lr_scheduler_config,
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        building_features, schedules, climates, targets = batch
        preds = self.model(building_features, schedules, climates)
        targets = targets.reshape(preds.shape)
        loss = F.mse_loss(preds, targets)
        self.log("Loss/Train", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Create a dictionary for tracking different residuals/metrics
        error_dict = {}
        seen_key = "SeenEPW" if dataloader_idx == 0 else "UnseenEPW"

        building_features, schedules, climates, targets = batch

        # targets are shaped (batch_size, n_zones x n_end_uses x n_timesteps)
        # preds are shaped (batch_size, n_zones x n_end_uses, n_timesteps)
        # targets are typically (b, 48)
        # preds are typically (b, 4, 12)
        # so some reshaping is done when necessary
        preds_hierarchical: torch.Tensor = self.model(
            building_features, schedules, climates
        )
        targets_hierarchical: torch.Tensor = targets.reshape(preds_hierarchical.shape)

        loss = F.mse_loss(preds_hierarchical, targets_hierarchical)

        error_dict[f"Loss/Val/{seen_key}"] = loss

        # transform preds and targets to energy units
        preds_energy = self.target_transform.inverse_transform(
            preds_hierarchical.reshape(targets.shape)
        ).reshape(preds_hierarchical.shape)

        targets_energy = self.target_transform.inverse_transform(targets).reshape(
            preds_hierarchical.shape
        )

        # Compute annual metrics
        annual_preds = preds_energy.sum(dim=-1)
        annual_targets = targets_energy.sum(dim=-1)
        annual_errors = torch.abs(annual_preds - annual_targets).mean(dim=0)
        annual_percent_errors = (
            torch.abs(annual_preds - annual_targets) / (annual_targets + 1)
        ).mean(dim=0) * 100

        # Store error metrics
        for i in range(annual_errors.shape[0]):
            slug = ""
            if i == 0:
                slug = "Core/Heating"
            elif i == 1:
                slug = "Core/Cooling"
            elif i == 2:
                slug = "Perimeter/Heating"
            elif i == 3:
                slug = "Perimeter/Cooling"
            error_dict[f"Error/Val/{seen_key}/{slug}"] = annual_errors[i]
            error_dict[f"PercentError/Val/{seen_key}/{slug}"] = annual_percent_errors[i]

        self.log_dict(error_dict, on_epoch=True)
        self.log(
            f"Loss/Val/{'SeenEPW' if dataloader_idx==0 else 'UnseenEPW'}",
            loss,
            on_epoch=True,
        )


if __name__ == "__main__":
    from ml.data import BuildingDataModule
    from pathlib import Path

    dm = BuildingDataModule(
        bucket="ml-for-bem",
        remote_experiment="full_climate_zone/v3",
        data_dir="data/lightning",
        climate_array_path=str(Path("data") / "epws" / "global_climate_array.npy"),
        batch_size=32,
    )
    # TODO: we should have a better workflow for first fitting the target transform
    # so that we can pass it into the modle.  I don't love that we have to manually call the hooks here
    dm.prepare_data()
    dm.setup(stage=None)
    target_transform = dm.target_transform

    # TODO: these should be inferred automatically from the datasets
    n_climate_timeseries = 7
    n_building_timeseries = 3
    timeseries_channels_per_input = n_climate_timeseries + n_building_timeseries
    static_features_per_input = 52
    timeseries_channels_per_output = 4
    timeseries_steps_per_output = 12

    """
    Hyperparameters:
    """
    lr = 1e-3
    lr_gamma = 0.5
    net_config = "Small"
    latent_factor = 4
    energy_cnn_feature_maps = 256
    energy_cnn_n_layers = 3
    energy_cnn_n_blocks = 6

    surrogate = Surrogate(
        lr=lr,
        lr_gamma=lr_gamma,
        target_transform=target_transform,
        net_config=net_config,
        latent_factor=latent_factor,
        energy_cnn_feature_maps=energy_cnn_feature_maps,
        energy_cnn_n_layers=energy_cnn_n_layers,
        energy_cnn_n_blocks=energy_cnn_n_blocks,
        timeseries_channels_per_input=timeseries_channels_per_input,
        static_features_per_input=static_features_per_input,
        timeseries_channels_per_output=timeseries_channels_per_output,
        timeseries_steps_per_output=timeseries_steps_per_output,
    )

    """
    Trainer
    """

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        enable_checkpointing=True,
        enable_model_summary=True,
        val_check_interval=0.05,
        limit_val_batches=0.3,
        num_sanity_val_steps=10,
        precision="16-mixed",
    )

    trainer.fit(
        model=surrogate,
        datamodule=dm,
    )
