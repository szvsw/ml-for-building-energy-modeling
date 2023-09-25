import os
from tqdm import tqdm

from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

from storage import upload_to_bucket

from schema import (
    Schema,
    ShoeboxGeometryParameter,
    BuildingTemplateParameter,
    WhiteboxSimulation,
    WindowParameter,
    SchedulesParameters,
    WINDOW_TYPES,
    ECONOMIZER_TYPES,
    RECOVERY_TYPES,
)
from nrel_uitls import ResStockConfiguration, CLIMATEZONES, RESTYPES
import logging

import wandb

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def sample(sample_name, GCS_UPLOAD_DIR):
    with wandb.init(
        job_type="sampling",
        project="ml-for-bem",
        tags=[
            "sampling",
            "data generation",
        ],
        group="sampling",
        notes="Mixed hierarchical sampling combining gridding with perturbations",
        name=sample_name,
        config={
            "GCS_UPLOAD_DIR": GCS_UPLOAD_DIR,
        },
    ) as run:
        schema = Schema()
        base_path = Path(os.path.dirname(os.path.dirname(__file__)))

        artifact = run.use_artifact("global-epw-data:latest")

        path = artifact.get_path("global_training_epw_meta.csv")
        loc = path.download()
        training_epw_metadata_df = pd.read_csv(loc)

        path = artifact.get_path("global_testing_epw_meta.csv")
        loc = path.download()
        testing_epw_metadata_df = pd.read_csv(loc)

        training_batch_ct = len(training_epw_metadata_df)
        training_epws_in_test_set_ct = 100
        testing_batch_ct = len(testing_epw_metadata_df) + training_epws_in_test_set_ct
        logger.info(
            f"{training_batch_ct} training batches will be generated, with each EPW appearing once."
        )
        logger.info(
            f"{testing_batch_ct} testing batches will be generated, with each testing EPW appearing once."
            "and 100 EPWs from the training set used as well."
        )
        total_batch_ct = training_batch_ct + testing_batch_ct
        for batch_id in tqdm(range(total_batch_ct)):
            is_training_sample = False
            is_test_sample_with_train_epw = False
            is_test_sample_with_test_epw = False
            if batch_id < training_batch_ct:
                is_training_sample = True
            elif batch_id < training_batch_ct + training_epws_in_test_set_ct:
                is_test_sample_with_train_epw = True
            else:
                is_test_sample_with_test_epw = True

            if is_training_sample or is_test_sample_with_train_epw:
                # Each epw exactly once in the train set,
                # and 100 of them in the test set.
                epw_counter = batch_id % training_batch_ct
                # Training EPW df is preshuffled, so this will choose
                epw_row = training_epw_metadata_df.iloc[epw_counter]
            else:
                assert is_test_sample_with_test_epw
                epw_counter = (
                    batch_id - training_batch_ct - training_epws_in_test_set_ct
                )
                epw_row = testing_epw_metadata_df.iloc[epw_counter]
            epw_idx = epw_row["idx"]
            epw_cz = epw_row["CZ"]
            if epw_cz == "7" or epw_cz == "8":
                epw_cz = epw_cz + "A"
            epw_cz_idx = CLIMATEZONES[epw_cz]

            storage_batch = schema.generate_empty_storage_batch(1)

            # Set Constants
            schema.update_storage_batch(
                storage_batch, parameter="batch_id", value=batch_id
            )
            schema.update_storage_batch(
                storage_batch, parameter="base_epw", value=epw_idx
            )
            schema.update_storage_batch(
                storage_batch, parameter="climate_zone", value=epw_cz_idx
            )

            # Pick an Archetype at random for this batch
            archetypes = list(RESTYPES.values())
            archetype = archetypes[np.random.randint(0, len(archetypes))]
            schema.update_storage_batch(
                storage_batch, parameter="program_type", value=archetype
            )

            # Pick a vintage at random
            vintage = np.random.randint(0, 4)
            schema.update_storage_batch(
                storage_batch, parameter="vintage", value=vintage
            )

            # Set Defaults
            schema.update_storage_batch(
                storage_batch,
                parameter="wwr",
                value=np.random.rand() * schema["wwr"].range + schema["wwr"].min,
            )

            # Pick Heating and Cooling Setpoints for this batch
            hsp = np.random.normal(loc=17.5, scale=2.5)
            csp = np.random.normal(loc=25, scale=2.5)

            schema.update_storage_batch(
                storage_batch, parameter="HeatingSetpoint", value=hsp
            )
            schema.update_storage_batch(
                storage_batch, parameter="CoolingSetpoint", value=csp
            )

            # Pick Infiltration baswe for the batch
            schema.update_storage_batch(
                storage_batch,
                parameter="Infiltration",
                value=np.random.rand() * schema["Infiltration"].range
                + schema["Infiltration"].min,
            )

            # Pick EPD/LPD/PD base for the batch
            schema.update_storage_batch(
                storage_batch,
                parameter="EquipmentPowerDensity",
                value=np.random.rand() * schema["EquipmentPowerDensity"].range
                + schema["EquipmentPowerDensity"].min,
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="LightingPowerDensity",
                value=np.random.rand() * schema["LightingPowerDensity"].range
                + schema["LightingPowerDensity"].min,
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="PeopleDensity",
                value=np.random.rand() * schema["PeopleDensity"].range
                + schema["PeopleDensity"].min,
            )

            # Assign construction props for the base of the batch
            base_facade_mass = (
                np.random.rand() * schema["FacadeMass"].range + schema["FacadeMass"].min
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="FacadeMass",
                value=base_facade_mass,
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="FacadeRValue",
                value=np.random.rand() * schema["FacadeRValue"].range
                + schema["FacadeRValue"].min,
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="RoofRValue",
                value=np.random.rand() * schema["RoofRValue"].range
                + schema["RoofRValue"].min,
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="SlabRValue",
                value=np.random.rand() * schema["SlabRValue"].range
                + schema["SlabRValue"].min,
            )

            schema.update_storage_batch(
                storage_batch,
                parameter="WindowSettings",
                value=np.random.choice(list(WINDOW_TYPES.keys())),
            )

            # TODO: roofmass
            # TODO: windows
            # TODO: add summed output of product of schedules with loads
            # TODO: Internal mass?

            # Generate Orientations
            storage_batch = storage_batch.repeat(4, axis=0)
            orientations = np.arange(4).reshape(-1, 1)
            schema.update_storage_batch(
                storage_batch, parameter="orientation", value=orientations
            )

            # geneate geometric variations
            geo_var_ct = 5
            storage_batch = storage_batch.repeat(geo_var_ct, axis=0)
            for j, parameter in enumerate(schema.parameters):
                if isinstance(parameter, ShoeboxGeometryParameter):
                    name = parameter.name
                    mean = parameter.mean
                    std = parameter.std
                    shape = parameter.shape_storage
                    values = np.random.normal(
                        loc=mean, scale=std, size=(storage_batch.shape[0], *shape)
                    )
                    schema.update_storage_batch(
                        storage_batch, parameter=name, value=values
                    )

            """Generate Schedules"""
            sched_var_ct = 10
            storage_batch = storage_batch.repeat(sched_var_ct, axis=0)
            for minibatch_ix in range(0, storage_batch.shape[0], sched_var_ct):
                vec_base_ix = minibatch_ix
                for sched_ix in range(sched_var_ct):
                    vec_ix = vec_base_ix + sched_ix
                    schedules = schema["schedules"].extract_storage_values(
                        storage_batch[vec_ix]
                    )
                    if sched_ix == 0:
                        # use default sched
                        pass
                    elif sched_ix == 1:
                        # all on
                        for j in range(3):
                            schedules[j, SchedulesParameters.op_indices["on/off"]] = 1
                    elif sched_ix == 2:
                        # all off
                        for j in range(3):
                            schedules[j, SchedulesParameters.op_indices["on/off"]] = -1
                    elif sched_ix == 3:
                        # Daily random
                        for j in range(3):
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_dilation"
                                ],
                            ] = 1  # 1hr per sample
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_samples"
                                ],
                            ] = 24  # 1 days worth of samples
                            schedules[
                                j, SchedulesParameters.op_indices["uniform_random_mode"]
                            ] = 0
                    elif sched_ix == 4:
                        # Daily random
                        for j in range(3):
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_dilation"
                                ],
                            ] = 1  # 1hr per sample
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_samples"
                                ],
                            ] = (
                                24 * 7
                            )  # 1 week worth of samples
                            schedules[
                                j, SchedulesParameters.op_indices["uniform_random_mode"]
                            ] = 2
                    elif sched_ix == 5:
                        # Daily rando
                        for j in range(3):
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_dilation"
                                ],
                            ] = (
                                j + 1
                            ) * 2  # 1hr per sample
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_samples"
                                ],
                            ] = (
                                24 / (j + 1) * 2 * 7
                            )  # 1 week worth of samples
                            schedules[
                                j, SchedulesParameters.op_indices["uniform_random_mode"]
                            ] = 2
                    # elif sched_ix == 4:
                    #     # Weekly random
                    #     for j in range(3):
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_dilation"]] = 4 # 4hr per sample
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_samples"]] = 24/4*7 # 1 week worth of samples
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_mode"]] = 0
                    # elif sched_ix == 5:
                    #     # Weekly random
                    #     for j in range(3):
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_dilation"]] = 2 # 2hr per sample
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_samples"]] = 24/2*7 # 2 worth of samples
                    #         schedules[j, SchedulesParameters.op_indices["uniform_random_mode"]] = 0
                    elif sched_ix == 6:
                        # Inverted
                        for j in range(3):
                            schedules[j, SchedulesParameters.op_indices["invert"]] = 1
                    elif sched_ix == 7:
                        # Noise
                        for j in range(3):
                            # schedules[j, SchedulesParameters.op_indices["noise"]] = 0.2
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_dilation"
                                ],
                            ] = 24  # 1day per sample
                            schedules[
                                j,
                                SchedulesParameters.op_indices[
                                    "uniform_random_samples"
                                ],
                            ] = 7  # 1 week worth of samples
                            schedules[
                                j, SchedulesParameters.op_indices["uniform_random_mode"]
                            ] = 0
                    elif sched_ix == 8:
                        # work week
                        for j in range(3):
                            schedules[
                                j, SchedulesParameters.op_indices["pulse_period"]
                            ] = (
                                24 * 7
                            )  # 1 week
                            schedules[
                                j, SchedulesParameters.op_indices["pulse_width"]
                            ] = (
                                24 * 5
                            )  # 5 days
                    elif sched_ix == 9:
                        # phasing
                        for j in range(3):
                            schedules[
                                j, SchedulesParameters.op_indices["pulse_period"]
                            ] = 12  # 1 week
                            schedules[
                                j, SchedulesParameters.op_indices["pulse_width"]
                            ] = 4  # 5 days
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_dilation"]] = int(np.random.random()*8+2) # 1hr per sample
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_samples"]] = 12
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_mode"]] = 0
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_dilation"]] = 21 # 21 hours per sample
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_samples"]] = 2*24*7 / 21 # 2 week worth of samples
                            # schedules[j, SchedulesParameters.op_indices["uniform_random_mode"]] = 0

            """Finalize Batch"""
            storage_batch = storage_batch.repeat(5, axis=0)
            n = storage_batch.shape[0]
            # TODO:? Appy specific upgrades? or just use perturbations

            """Pertub Geometry"""
            for j, parameter in enumerate(schema.parameters):
                if isinstance(parameter, ShoeboxGeometryParameter):
                    name = parameter.name
                    mean = parameter.mean
                    std = parameter.std
                    shape = parameter.shape_storage
                    values = schema[name].extract_storage_values_batch(storage_batch)
                    mods = np.random.normal(loc=0, scale=std / 4, size=(n, *shape))
                    values = values + mods
                    schema.update_storage_batch(
                        storage_batch, parameter=name, value=values
                    )

            """Perturb Building Parameters"""

            heating_setpoints = schema["HeatingSetpoint"].extract_storage_values_batch(
                storage_batch
            )
            shape = schema["HeatingSetpoint"].shape_storage
            heating_setpoints = heating_setpoints + np.random.normal(
                loc=0, scale=1, size=(n, *shape)
            )
            schema.update_storage_batch(
                storage_batch, parameter="HeatingSetpoint", value=heating_setpoints
            )

            shape = schema["CoolingSetpoint"].shape_storage
            cooling_setpoints = schema["CoolingSetpoint"].extract_storage_values_batch(
                storage_batch
            )
            cooling_setpoints = cooling_setpoints + np.random.normal(
                loc=0, scale=1, size=(n, *shape)
            )
            schema.update_storage_batch(
                storage_batch, parameter="CoolingSetpoint", value=cooling_setpoints
            )

            shape = schema["Infiltration"].shape_storage
            infiltration = schema["Infiltration"].extract_storage_values_batch(
                storage_batch
            )
            infiltration = infiltration + np.random.normal(
                loc=-0.2, scale=0.2, size=(n, *shape)
            )
            schema.update_storage_batch(
                storage_batch, parameter="Infiltration", value=infiltration
            )

            shape = schema["EquipmentPowerDensity"].shape_storage
            epd = schema["EquipmentPowerDensity"].extract_storage_values_batch(
                storage_batch
            )
            epd = epd + np.random.normal(loc=0, scale=2, size=(n, *shape))
            schema.update_storage_batch(
                storage_batch, parameter="EquipmentPowerDensity", value=epd
            )

            shape = schema["LightingPowerDensity"].shape_storage
            lpd = schema["LightingPowerDensity"].extract_storage_values_batch(
                storage_batch
            )
            lpd = lpd + np.random.normal(loc=1, scale=2, size=(n, *shape))
            schema.update_storage_batch(
                storage_batch, parameter="LightingPowerDensity", value=lpd
            )

            shape = schema["PeopleDensity"].shape_storage
            ppd = schema["PeopleDensity"].extract_storage_values_batch(storage_batch)
            ppd = ppd + np.random.normal(loc=0, scale=0.005, size=(n, *shape))
            schema.update_storage_batch(
                storage_batch, parameter="PeopleDensity", value=ppd
            )

            shape = schema["FacadeMass"].shape_storage
            facade_mass = schema["FacadeMass"].extract_storage_values_batch(
                storage_batch
            )
            facade_mass = facade_mass + np.random.normal(
                loc=0 if base_facade_mass > 50000 else 50000,
                scale=25000,
                size=(n, *shape),
            )
            schema.update_storage_batch(
                storage_batch, parameter="FacadeMass", value=facade_mass
            )

            shape = schema["RoofMass"].shape_storage
            roof_mass = np.random.normal(
                loc=schema["RoofMass"].mean,
                scale=schema["RoofMass"].std,
                size=(n, *shape),
            )
            schema.update_storage_batch(
                storage_batch, parameter="RoofMass", value=roof_mass
            )

            shape = schema["FacadeRValue"].shape_storage
            facade_rv = schema["FacadeRValue"].extract_storage_values_batch(
                storage_batch
            )
            facade_rv = facade_rv + np.random.normal(
                loc=1.5, scale=0.75, size=(n, *shape)
            )
            schema.update_storage_batch(
                storage_batch, parameter="FacadeRValue", value=facade_rv
            )

            shape = schema["RoofRValue"].shape_storage
            roof_rv = schema["RoofRValue"].extract_storage_values_batch(storage_batch)
            roof_rv = roof_rv + np.random.normal(loc=1.5, scale=0.75, size=(n, *shape))
            schema.update_storage_batch(
                storage_batch, parameter="RoofRValue", value=roof_rv
            )

            shape = schema["SlabRValue"].shape_storage
            slab_rv = schema["SlabRValue"].extract_storage_values_batch(storage_batch)
            slab_rv = slab_rv + np.random.normal(loc=0.5, scale=2, size=(n, *shape))
            schema.update_storage_batch(
                storage_batch, parameter="SlabRValue", value=slab_rv
            )

            # Windows
            window_vals = schema["WindowSettings"].extract_storage_values_batch(
                storage_batch
            )
            window_vals = np.where(
                np.random.random((window_vals.shape)) > 0.5,
                np.random.randint(
                    0,
                    len(WINDOW_TYPES),
                    size=window_vals.shape,
                ),
                window_vals,
            )
            schema.update_storage_batch(
                storage_batch, parameter="WindowSettings", value=window_vals
            )

            # Protect Setpoints
            heating_setpoints = schema["HeatingSetpoint"].extract_storage_values_batch(
                storage_batch
            )
            cooling_setpoints = schema["CoolingSetpoint"].extract_storage_values_batch(
                storage_batch
            )
            heating_setpoints = np.where(
                heating_setpoints > cooling_setpoints,
                cooling_setpoints - 2,
                heating_setpoints,
            )
            schema.update_storage_batch(
                storage_batch, parameter="HeatingSetpoint", value=heating_setpoints
            )

            schema.update_storage_batch(
                storage_batch,
                parameter="shading_seed",
                value=np.random.randint(0, 20000, size=(window_vals.shape)),
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="EconomizerSettings",
                value=np.random.randint(
                    0, len(ECONOMIZER_TYPES), size=(window_vals.shape)
                ),
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="RecoverySettings",
                value=np.random.randint(
                    0, len(RECOVERY_TYPES), size=(window_vals.shape)
                ),
            )

            # Set Variation ID
            variation_ids = np.arange(n)
            schema.update_storage_batch(
                storage_batch, parameter="variation_id", value=variation_ids
            )
            schema.update_storage_batch(
                storage_batch,
                parameter="schedules_seed",
                value=(np.random.random(1)) * n * variation_ids,
            )

            # Write to an HDF5 file
            slug = f"BATCH_{batch_id:05d}.hdf5"
            if is_training_sample:
                slug = f"train_epws_train_set/{slug}"
            elif is_test_sample_with_train_epw:
                slug = f"train_epws_in_test_set/{slug}"
            elif is_test_sample_with_test_epw:
                slug = f"test_epws_in_test_set/{slug}"
            outfile = f"./data/hdf5/batch_v2/{slug}"
            with h5py.File(outfile, "w") as f:
                f.create_dataset(
                    name="storage_vectors",
                    shape=storage_batch.shape,
                    dtype=storage_batch.dtype,
                    data=storage_batch,
                )

            # upload to cloud bucket for easy backup
            destination = f"{wandb.config.GCS_UPLOAD_DIR}/{slug}"
            upload_to_bucket(blob_name=destination, file_name=outfile)

        art = wandb.Artifact(
            name="sample-storage-batches",
            type="dataset",
            metadata={
                "count": "1000 samples per batch",
                "layout": "row",
                "info": "Each batch is a single epw.  "
                "The train set has no repeating epws between batches. "
                "100 EPWs from the train set appear in the test set. "
                "The remaining 100 batches have completely unseen EPWs. ",
            },
        )
        art.add_reference(f"gs://ml-for-bem-data/{wandb.config.GCS_UPLOAD_DIR}")
        run.log_artifact(art)


def train_test_split_epw(count=4, seed=42):
    with wandb.init(
        project="ml-for-bem",
        name="split-epws-for-training",
        job_type="data-preparation",
        tags=["weather"],
        group="sampling",
        notes="Splits weather files into two groups, one which will "
        "appear in both training and testing, and one which will only "
        "appear in testing.",
        config={
            "count": count,
            "seed": seed,
        },
        save_code=True,
    ) as run:
        # Get the list of ~850 Cities and download
        art = run.use_artifact("global-epw-data:latest")
        path = art.get_path("global_epws_indexed.csv")
        loc = path.download()

        # Create a new draft document
        art = art.new_draft()

        # Load the data
        df = pd.read_csv(loc)

        # Drop unecessary columns
        df = df.drop(labels=(col for col in df.columns if "Unnamed" in col), axis=1)

        # Get the climate zone possibilities
        czs = df["CZ"].unique()

        # set a random seed
        np.random.seed(wandb.config.seed)

        # make a new data frame for the cities to pop for testing
        withheld_city_df = pd.DataFrame()

        for cz in czs:
            # downsample to this climate zone
            cz_df = df[df["CZ"] == cz]
            # If a climate zone has fewer than 5 cities, we will skip it
            # and leave it in training
            if len(cz_df) < 5:
                continue

            # Pick some integers for climate zones to pop
            # The number to pop is part of the Run Config
            withheld_ix = np.random.choice(
                range(len(cz_df)), size=(wandb.config.count), replace=False
            )

            # Get the cities
            withheld_cities = cz_df.iloc[withheld_ix]

            # Append them to the list of withheld cities
            withheld_city_df = pd.concat((withheld_city_df, withheld_cities), axis=0)

            # and then pop them from the list of training cities
            df = df.drop(labels=cz_df.iloc[withheld_ix].index, axis=0)

            # make sure we did it successfully
            for cit in withheld_city_df["city"]:
                assert cit not in df["city"]

        # Shuffle the order of the dataframes that way
        # we can just iterate straight through them when sampling
        new_order = np.random.permutation(len(df))
        df = df.iloc[new_order]

        new_order = np.random.permutation(len(withheld_city_df))
        withheld_city_df = withheld_city_df.iloc[new_order]

        # Save
        df.to_csv(
            "./data/epws/global_training_epw_meta.csv",
        )
        withheld_city_df.to_csv(
            "./data/epws/global_testing_epw_meta.csv",
        )

        # Upload
        upload_to_bucket(
            blob_name="weather/global_training_epw_meta.csv",
            file_name="./data/epws/global_training_epw_meta.csv",
        )
        upload_to_bucket(
            blob_name="weather/global_testing_epw_meta.csv",
            file_name="./data/epws/global_testing_epw_meta.csv",
        )
        art.remove("global_training_epw_meta.csv")
        art.remove("global_testing_epw_meta.csv")

        # Log
        art.add_reference("gs://ml-for-bem-data/weather")
        run.log_artifact(art)


if __name__ == "__main__":
    from storage import config_gcs_adc

    config_gcs_adc()
    sample(sample_name="SAMPLE_BATCH_2023_SEPT", GCS_UPLOAD_DIR="SAMPLES_2023")
