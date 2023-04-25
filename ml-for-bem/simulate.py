import logging
import os
import sys
import time

import numpy as np
import h5py

from pathlib import Path

from archetypal import parallel_process

from storage import upload_to_bucket
from schema import Schema, WhiteboxSimulation

logging.basicConfig()
logger = logging.getLogger("Batch Simulator")
logger.setLevel(logging.INFO)


class BatchSimulator:
    __slots__ = (
        "batch_id",
        "storage_batch",
        "batch_size",
        "processes",
        "parallel_config",
        "schema",
    )

    def __init__(self, schema, batch_id, processes):
        self.batch_id = batch_id
        self.processes = processes
        self.schema = schema
        logger.info("--------- Batch Simulation ---------")
        logger.info(f"Batch ID: {self.batch_id}")
        logger.info(f"Opening HDF5 storage batch file for batch {self.batch_id}...")
        with h5py.File(self.storage_batch_filepath, "r") as f:
            self.storage_batch = f["storage_vectors"][...]

        self.batch_size = self.storage_batch.shape[0]
        logger.info(
            f"Loaded BATCH:{self.batch_id}, which has {self.batch_size} design vectors"
        )
        self.construct_parallel_process_dict()

    @property
    def storage_batch_filepath(self):
        return (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "data"
            / "hdf5"
            / f"batch_{self.batch_id:05d}.hdf5"
        )

    @property
    def results_batch_filepath(self):
        return (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "data"
            / "hdf5"
            / f"batch_{self.batch_id:05d}_results.hdf5"
        )

    def construct_parallel_process_dict(self):
        logger.info(f"Building parallel dict config for BATCH:{self.batch_id}...")
        self.parallel_config = {
            i: {"idx": i, "storage_vector": self.storage_batch[i, :]}
            for i in range(self.batch_size)
        }

    def simulate(self, idx, storage_vector):
        whitebox = WhiteboxSimulation(Schema(), storage_vector)
        res_hourly, res_monthly = whitebox.simulate()
        return {"hourly": res_hourly, "monthly": res_monthly}

    def run(self):
        logger.info(
            f"Launching parallel processing for BATCH:{self.batch_id} with {self.processes} p.processors..."
        )
        results = parallel_process(
            in_dict=self.parallel_config,
            function=self.simulate,
            use_kwargs=True,
            processors=self.processes,
        )
        logger.info(
            f"Completed parallel processing for BATCH:{self.batch_id} with {self.processes}!"
        )
        # TODO: consider more hdf5 styled data transfer, pd.to_hdf5 method, etc
        logger.info(f"Moving data to numpy tensors for BATCH:{self.batch_id}!")
        results_hourly_tensor = np.zeros(
            shape=(self.batch_size, *self.schema.sim_output_shape)
        )
        results_monthly_tensor = np.zeros(
            shape=(self.batch_size, self.schema.sim_output_shape[0], 12)
        )
        for ix, result in results.items():
            try:
                results_hourly_tensor[ix] = result["hourly"].to_numpy(dtype=np.float32).T
                results_monthly_tensor[ix] = result["monthly"].to_numpy(dtype=np.float32).T
            except TypeError:
                logger.error(f"No simulation data found for BATCH:{self.batch_id}, INDEX:{ix}")
                # TODO: Better error handling
                results_hourly_tensor[ix,:,:] = -1
                results_monthly_tensor[ix,:,:] = -1


        logger.info(f"Writing results to HDF5 for BATCH:{self.batch_id}...")
        with h5py.File(self.results_batch_filepath, "w") as f:
            # TODO: consider adding metadata labels to files which explain tensor shape
            f.create_dataset(
                "hourly",
                shape=results_hourly_tensor.shape,
                data=results_hourly_tensor,
                compression="gzip",
                compression_opts=6,
            )
            f.create_dataset(
                "monthly",
                shape=results_hourly_tensor.shape,
                data=results_hourly_tensor,
                compression="gzip",
                compression_opts=6,
            )
        logger.info(f"Done writing results to HDF5 for BATCH:{self.batch_id}.")

    def upload(self):
        logger.info(f"Upload results to GCP bucket for BATCH:{self.batch_id}...")
        slug = os.path.split(self.results_batch_filepath)[-1]
        upload_to_bucket(f"results/{slug}", self.results_batch_filepath)
        logger.info(f"Done uploading results to GCP bucket for BATCH:{self.batch_id}.")


if __name__ == "__main__":
    batch_id = int(sys.argv[1])
    processes = int(sys.argv[2])
    schema = Schema()
    batch = BatchSimulator(schema=schema, batch_id=batch_id, processes=processes)
    batch.run()
    batch.upload()
