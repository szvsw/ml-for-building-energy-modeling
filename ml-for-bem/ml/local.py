from pathlib import Path
from typing import Literal, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from ml.data import MinMaxTransform, StdNormalTransform, WeatherStdNormalTransform
from ml.networks import MLP



# TODO: Surrogate should have a `forward`` method


class LocalSurrogate(pl.LightningModule):
    """
    A PyTorch Lightning abstraction for managing a surrogate model which can predict energy from building features for a single EPW
    
    """

    def __init__(
        self,
        target_transform: Union[MinMaxTransform, StdNormalTransform],
        space_config: dict,
        lr: float = 1e-3,
        lr_gamma: float = 0.5,
        input_dim: int = 50,
        hidden_dim: int = 100,
        block_depth: int = 3,
        block_count: int = 12,
        output_dim: int = 48,
        dropout: float = 0.3,
        activation: Literal["SELU", "SiLU", "ReLU", "LeakyRELU", "GELU"] = "SiLU"
    ):
        """
        Create the Lighting Module for managing the local surrogate

        Args:
            target_transform (nn.Module): a transform to apply to targets.  Should implement the `inverse_transform` method.
            space_config (dict): the configuration of the design space
            lr (float, optional): the learning rate.  Defaults to 1e-3.
            lr_gamma (float, optional): the learning rate decay.  Defaults to 0.5. Called after each epoch completes.
            latent_factor (int, optional): The timeseries net will output a latent representation with `latent_factor * static_features_per_input` channels.  Defaults to 4.
            input_dim (int, optional): dimensionality of the input
            hidden_dim (int, optional): width of hidden layers in each skip block
            block_depth (int, optional): number of hidden layers per skip block
            block_count (int, optional): number of skip blocks in network
            output_dim (int, optional): dimensionality of the output
            dropout (float, optional): dropout to apply for regularization
            activation (str): Which activation to use


        Returns:
            Surrogate: the PyTorch Lightning Module which manages the surrogate model

        """
        # TODO: store features transform, not just config
        super().__init__()

        # Store all the hyperparameters
        self.space_config = space_config
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.block_depth = block_depth
        self.block_count = block_count
        self.output_dim = output_dim
        self.dropout=dropout
        

        # Save hyperparameters
        self.save_hyperparameters()

        # Create the net
        self.model = MLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            block_depth=self.block_depth,
            block_count = block_count,
            output_dim=self.output_dim,
            dropout=self.dropout
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
            "lr_scheduler": lr_scheduler_config,
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        features, targets = batch
        preds = self.model(features)
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
        annual_residuals = annual_preds - annual_targets
        annual_mse = annual_residuals.pow(2).mean(dim=0)
        annual_rmse = torch.sqrt(annual_mse)
        annual_cvrmse = annual_rmse / annual_targets.mean(dim=0) * 100
        annual_mae = torch.abs(annual_residuals).mean(dim=0)
        annual_symmetric_mape = (
            torch.abs(annual_residuals)
            / ((torch.abs(annual_targets) + torch.abs(annual_preds) + 1e-3) / 2)
        ).mean(dim=0) * 100
        annual_mape = (
            torch.abs(annual_residuals) / (torch.abs(annual_targets) + 1e-3)
        ).mean(dim=0) * 100
        monthly_mse = F.mse_loss(preds_energy, targets_energy)
        monthly_mae = torch.abs(preds_energy - targets_energy).mean()
        monthly_rmse = torch.sqrt(monthly_mse)
        monthly_cvrmse = monthly_rmse / targets_energy.mean() * 100
        error_dict[f"MonthlyMAE/Val/{seen_key}"] = monthly_mae
        error_dict[f"MonthlyMSE/Val/{seen_key}"] = monthly_mse
        error_dict[f"MonthlyRMSE/Val/{seen_key}"] = monthly_rmse
        error_dict[f"MonthlyCVRMSE/Val/{seen_key}"] = monthly_cvrmse

        # Store error metrics
        for i in range(annual_mae.shape[0]):
            slug = ""
            if i == 0:
                slug = "Core/Heating"
            elif i == 1:
                slug = "Core/Cooling"
            elif i == 2:
                slug = "Perimeter/Heating"
            elif i == 3:
                slug = "Perimeter/Cooling"
            error_dict[f"AnnualMAE/Val/{seen_key}/{slug}"] = annual_mae[i]
            error_dict[f"AnnualMSE/Val/{seen_key}/{slug}"] = annual_mae[i]
            error_dict[f"AnnualMAPE/Val/{seen_key}/{slug}"] = annual_mape[i]
            error_dict[f"AnnualMAPESYM/Val/{seen_key}/{slug}"] = annual_symmetric_mape[
                i
            ]
            error_dict[f"AnnualRMSE/Val/{seen_key}/{slug}"] = annual_rmse[i]
            error_dict[f"AnnualCVRMSE/Val/{seen_key}/{slug}"] = annual_cvrmse[i]
        self.log_dict(
            error_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"Loss/Val/{seen_key}",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, schedules, climates = batch
        climates = self.weather_transform(climates)
        preds = self.model(features, schedules, climates)
        preds = preds.reshape(features.shape[0], -1)
        preds = self.target_transform.inverse_transform(preds)
        return preds

    @classmethod
    def load_from_registry(
        cls,
        registry="ml-for-building-energy-modeling/model-registry",
        model: str = "Global UBEM Shoebox Surrogate with Combined TS Embedder",
        tag: str = "latest",
        resource: str = "model.ckpt",
    ) -> "Surrogate":
        """
        Fetches a surrogate model from the W&B cloud.

        Args:
            registry (str): The W&B registry to fetch the model from.
            model (str): The model name.
            tag (str): The model tag.
            resource (str): The file resource to fetch from within the model artifact.

        Returns:
            surrogate (Surrogate): The surrogate model.
        """
        api = wandb.Api()
        local_dir = Path("data") / "models" / tag
        model_str = f"{registry}/{model}:{tag}"
        surrogate_artifact = api.artifact(model_str, type="model")
        pth = surrogate_artifact.get_path(resource)
        model_path = pth.download(local_dir)
        surrogate = cls.load_from_checkpoint(model_path)
        return surrogate


if __name__ == "__main__":
    import os

    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

    from ml.data import BuildingDataModule

    # TODO: batch size should be in config
    # TODO: thresh should be in config
    wandb.login()
    in_lightning_studio = (
        True if os.environ.get("LIGHTNING_ORG", None) is not None else False
    )
    bucket = "ml-for-bem"
    remote_experiment = "full_climate_zone/v7"
    climate_experiment = "weather/v1"

    local_data_dir = (
        "/teamspace/s3_connections/ml-for-bem"
        if in_lightning_studio
        else "data/lightning"
    )
    remote_data_dir = "full_climate_zone/v7/lightning"
    remote_data_path = f"s3://{bucket}/{remote_data_dir}"

    dm = BuildingDataModule(
        bucket=bucket,
        remote_experiment=remote_experiment,
        data_dir=local_data_dir,
        climate_experiment=climate_experiment,
        batch_size=128,
        val_batch_mult=4,
    )
    # TODO: we should have a better workflow for first fitting the target transform
    # so that we can pass it into the modle.  I don't love that we have to manually call the hooks here
    # TODO: the model should store the climate transform, or we should otherwise have a better way of
    # storing it for use later.
    dm.prepare_data()
    dm.setup(stage=None)
    target_transform = dm.target_transform
    weather_transform = dm.weather_transform
    space_config = dm.space_config

    # TODO: these should be inferred automatically from the datasets
    n_climate_timeseries = len(weather_transform.channel_names)
    n_building_timeseries = 3
    timeseries_channels_per_input = n_climate_timeseries + n_building_timeseries
    static_features_per_input = 49
    timeseries_channels_per_output = 4
    timeseries_steps_per_output = 12

    """
    Hyperparameters:
    """
    lr = 1e-2  # TODO: larger learning rate for larger batch size on multi-gpu?
    lr_gamma = 0.95
    net_config = "Small"
    latent_factor = 4
    energy_cnn_feature_maps = 512
    energy_cnn_n_layers = 3
    energy_cnn_n_blocks = 12

    surrogate = Surrogate(
        lr=lr,
        lr_gamma=lr_gamma,
        target_transform=target_transform,
        weather_transform=weather_transform,
        space_config=space_config,
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
    # surrogate = Surrogate.load_from_checkpoint("data/models/model.ckpt")
    
    """
    Loggers
    """
    wandb_logger = WandbLogger(
        project="ml-for-bem",
        name="Surrogate-With-Solar-Position",
        save_dir="wandb",
        log_model="all",
        job_type="train",
        group="global-surrogate",
    )

    """
    Trainer
    """

    # TODO: We should have better model checkpointing so we
    # don't blow up the disk and can better track the best model
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        # strategy="auto",
        strategy="ddp_find_unused_parameters_true",
        logger=wandb_logger,
        default_root_dir=remote_data_path,
        enable_progress_bar=True,
        enable_checkpointing=True,
        enable_model_summary=True,
        val_check_interval=0.25,
        # check_val_every_n_epoch=1,
        num_sanity_val_steps=3,
        precision="bf16-mixed",
        # gradient_clip_val=0.5,
        sync_batchnorm=True,
    )

    trainer.fit(
        model=surrogate,
        datamodule=dm,
        # ckpt_path="data/models/model.ckpt"
    )
