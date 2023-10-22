from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ml.networks import EnergyCNN2, ConvNet, Conv1DStageConfig
from ml.data import MinMaxTransform


class MultiModalModel(nn.Module):
    def __init__(self, timeseries_net: nn.Module, energy_net: nn.Module):
        super().__init__()
        self.timeseries_net = timeseries_net
        self.energy_net = energy_net

    def forward(
        self,
        building_features: torch.Tensor,
        climates: torch.Tensor,
        schedules: torch.Tensor,
    ) -> torch.Tensor:
        timeseries = torch.cat([climates, schedules], dim=1)
        latent = self.timeseries_net(timeseries)
        building_features = building_features.unsqueeze(-1)
        building_features = building_features.repeat(1, 1, latent.shape[-1])
        input = torch.cat([building_features, latent], dim=1)
        preds = self.energy_net(input)
        return preds


class Surrogate(pl.LightningModule):
    def __init__(
        self,
        target_transform: MinMaxTransform,
        lr: float = 1e-3,
        latent_factor: int = 4,
        energy_cnn_feature_maps: int = 128,
        energy_cnn_n_layers: int = 3,
        timeseries_channels_per_input: int = 10,
        static_features_per_input: int = 10,
        timeseries_channels_per_output: int = 4,
        timeseries_steps_per_output: int = 12,
    ):
        # TODO: auto configure network dims
        # TODO: hyperparameters for configure block architectures
        super().__init__()
        self.lr = lr
        self.timeseries_channels_per_input = timeseries_channels_per_input
        self.static_features_per_input = static_features_per_input
        self.latent_factor = latent_factor
        self.latent_channels = self.static_features_per_input * self.latent_factor
        self.energy_cnn_in_size = self.latent_channels + self.static_features_per_input
        self.timeseries_channels_per_output = timeseries_channels_per_output
        self.timeseries_steps_per_output = timeseries_steps_per_output
        self.energy_cnn_feature_maps = energy_cnn_feature_maps
        self.energy_cnn_n_layers = energy_cnn_n_layers

        conf = Conv1DStageConfig.Base(self.timeseries_channels_per_input)

        timeseries_net = ConvNet(
            stage_configs=conf,
            latent_channels=self.latent_channels,
            latent_length=self.timeseries_steps_per_output,
        )

        energy_net = EnergyCNN2(
            in_channels=self.energy_cnn_in_size,
            out_channels=self.timeseries_channels_per_output,
            n_feature_maps=self.energy_cnn_feature_maps,
            n_layers=self.energy_cnn_n_layers,
        )
        self.target_transform = target_transform

        self.model: MultiModalModel = MultiModalModel(
            timeseries_net=timeseries_net, energy_net=energy_net
        )

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        building_features, schedules, climates, targets = batch
        preds = self.model(building_features, schedules, climates)
        targets = targets.reshape(preds.shape)
        loss = F.mse_loss(preds, targets)
        self.log("Loss/Train", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        error_dict = {}
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
        error_dict[f"Loss/Val/{'SeenEPW' if dataloader_idx==0 else 'UnseenEPW'}"] = loss

        # transform preds and targets to energy units
        preds_energy = self.target_transform.inverse_transform(
            preds_hierarchical.reshape(targets.shape)
        ).reshape(preds_hierarchical.shape)
        targets_energy = self.target_transform.inverse_transform(targets).reshape(
            preds_hierarchical.shape
        )
        annual_preds = preds_energy.sum(dim=-1)
        annual_targets = targets_energy.sum(dim=-1)
        annual_errors = torch.abs(annual_preds - annual_targets).mean(dim=0)

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
            error_dict[
                f"Error/Val/{'SeenEPW' if dataloader_idx==0 else 'UnseenEPW'}/{slug}"
            ] = annual_errors[i]

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
        batch_size=64,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
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
    lr = 5e-5
    latent_factor = 7
    energy_cnn_feature_maps = 256
    energy_cnn_n_layers = 10

    surrogate = Surrogate(
        lr=lr,
        target_transform=target_transform,
        latent_factor=latent_factor,
        energy_cnn_feature_maps=energy_cnn_feature_maps,
        energy_cnn_n_layers=energy_cnn_n_layers,
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
        enable_checkpointing=False,
        enable_model_summary=True,
        val_check_interval=0.1,
        limit_val_batches=0.1,
        num_sanity_val_steps=10,
        precision="16-mixed",
    )

    trainer.fit(
        model=surrogate,
        datamodule=dm,
    )
