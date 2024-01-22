import calendar
import copy
import json
import logging
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from tqdm import tqdm
from utils.constants import *
from utils.schedules import mutate_timeseries

logging.basicConfig()
logger = logging.getLogger("EPJSON")
logger.setLevel(logging.DEBUG)


class EpJsonIDF:
    """
    A class for editing IDF files as EpJSONs (no archetypal dependence)
    """

    def __init__(
        self, idf_path, output_directory=None, eplus_loc=Path("C:\EnergyPlusV22-2-0")
    ):
        # get idf JSON
        self.eplus_location = Path(eplus_loc)
        self.idf_path = Path(idf_path)
        if output_directory:
            self.output_directory = output_directory
        else:
            self.output_directory = self.idf_path.parent

        self.epjson_path = self.convert(
            str(self.idf_path),
            self.eplus_location,
            str(self.output_directory),
            file_type="epjson",
        )
        with open(self.epjson_path, "r") as f:
            epjson = json.load(f)
            self.epjson = copy.deepcopy(epjson)

    @classmethod
    def convert(cls, path, eplus_location, output_directory, file_type="epjson"):
        logger.info(f"Converting {path} to {file_type}")
        # Define the command and its arguments
        cmd = eplus_location / f"energyplus{'.exe' if os.name == 'nt' else ''}"
        logger.debug(cmd)
        args = ["--convert-only", "--output-directory", output_directory, path]
        logger.debug(args)

        # TODO change location of idf

        # Run the command
        with subprocess.Popen(
            [cmd] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as proc:
            for line in proc.stdout:
                logger.info(line.strip())
            exit_code = proc.wait()

        # Check if the command was successful
        if exit_code == 0:
            logger.info("Command executed successfully.")
        else:
            logger.error(f"Command failed with exit code {exit_code}.")
            raise RuntimeError(f"Failed to convert EpJSON to IDF.")

        return str(path).split(".")[0] + f".{file_type}"

    def save(self):
        with open(self.epjson_path, "w") as f:
            json.dump(self.epjson, f, indent=4)

    def save_as(self, path):
        with open(path, "w") as f:
            json.dump(self.epjson, f, indent=4)

    def zone_update(self, key, zone_dict):
        for zone in self.epjson[key].values():
            zone.update(zone_dict)

    def save_idf(self, name=None, suffix=None, output_path=None):
        if output_path is None:
            output_path = self.output_directory
        if name:
            path = self.output_directory / name + ".epjson"
            self.save_as(path)
        elif suffix:
            path = str(self.epjson_path)[:-7] + suffix + ".epjson"
            self.save_as(path)
        else:
            path = self.epjson_path
            self.save()
        logger.info(f"Building idf for {path}")

        idf_path = self.convert(path, self.eplus_location, output_path, file_type="idf")
        return idf_path

    @classmethod
    def day_to_epbunch(cls, dsched, idx=0, sched_lim=sched_type_limits):
        return {
            dsched.Name: dict(
                **{"hour_{}".format(i + 1): dsched.all_values[i] for i in range(24)},
                schedule_type_limits_name=sched_lim["Name"],
            )
        }

    @classmethod
    def week_to_epbunch(cls, wsched, idx=0, sched_lim=sched_type_limits):
        return {
            wsched.Name: dict(
                **{
                    f"{calendar.day_name[i].lower()}_schedule_day_name": day.Name
                    for i, day in enumerate(wsched.Days)
                },
                holiday_schedule_day_name=wsched.Days[6].Name,
                summerdesignday_schedule_day_name=wsched.Days[0].Name,
                winterdesignday_schedule_day_name=wsched.Days[0].Name,
                customday1_schedule_day_name=wsched.Days[1].Name,
                customday2_schedule_day_name=wsched.Days[6].Name,
            )
        }

    @classmethod
    def year_to_epbunch(cls, sched, sched_lim=sched_type_limits):
        dict_list = []
        for i, part in enumerate(sched.Parts):
            dict_list.append(
                dict(
                    **{
                        "schedule_week_name": part.Schedule.Name,
                        "start_month": part.FromMonth,
                        "start_day".format(i + 1): part.FromDay,
                        "end_month".format(i + 1): part.ToMonth,
                        "end_day".format(i + 1): part.ToDay,
                    }
                )
            )
        return dict(
            schedule_type_limits_name=sched_lim["Name"],
            schedule_weeks=dict_list,
        )

    @classmethod
    def schedule_to_epbunch(cls, name, values, sched_lims_bunch=sched_type_limits):
        assert len(values) == 8760, "Schedule length does not equal 8760 hours!"
        arch_schedule = Schedule(Name=name, Values=values)
        y, w, d = arch_schedule.to_year_week_day()
        year_bunch = year_to_epbunch(y, sched_lims_bunch)
        week_bunches = []
        day_bunches = []
        for week in w:
            week_bunches.append(week_to_epbunch(week, sched_lims_bunch))
        for day in d:
            day_bunches.append(day_to_epbunch(day, sched_lims_bunch))
        return year_bunch, week_bunches, day_bunches


def validation_ventilation(epjson, mech_vent_sched_mode, new_filename):
    if mech_vent_sched_mode == MechVentMode.OccupancySchedule.value:
        pass
    elif mech_vent_sched_mode == MechVentMode.AllOn.value:
        epjson.zone_update(
            "DesignSpecification:OutdoorAir", {"outdoor_air_schedule_name": "AllOn"}
        )
        epjson.zone_update(
            "ZoneHVAC:IdealLoadsAirSystem",
            {"demand_controlled_ventilation_type": "None"},
        )
    else:
        epjson.zone_update(
            "DesignSpecification:OutdoorAir", {"outdoor_air_schedule_name": "Off"}
        )
        epjson.zone_update(
            "ZoneHVAC:IdealLoadsAirSystem",
            {"demand_controlled_ventilation_type": "None"},
        )
    epjson.save_idf(output_path=epjson.output_directory.parent / new_filename)
    # epjson.save_idf(suffix="_new") # original_name_new.idf instead of override


def process_idfs(dat):
    idf_path = dat[0]
    mech_vent_sched_mode = dat[1]
    fname = dat[2]
    epjson = EpJsonIDF(idf_path)
    validation_ventilation(epjson, mech_vent_sched_mode, fname)
    os.remove(epjson.epjson_path)
    return idf_path


def set_validation_ventilation_schedules(hdf, fname):
    local_dir = Path(hdf).parent
    features = pd.read_hdf(hdf, key="buildings")
    names = features.index.to_list()
    vent_modes = features["VentilationMode"].to_list()
    idf_name = lambda x: "%09d" % (int(x),) + ".idf"
    idf_paths = [local_dir / "idf" / idf_name(x) for x in names]
    run_dict = [[x, y, fname] for x, y in zip(idf_paths, vent_modes)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        dfs = list(tqdm(executor.map(process_idfs, run_dict), total=len(run_dict)))
    logger.info("Downloading and opening files complete.")

    # delete extra files
    os.remove(local_dir / "idf" / "eplusout.end")
    os.remove(local_dir / "idf" / "eplusout.err")
    os.remove(local_dir / fname / "eplusout.end")
    os.remove(local_dir / fname / "eplusout.err")


# make a click function which accepts a number of simulations to run, a bucket name, and an experiment name
@click.command()
@click.option(
    "--hdf", default=None, help=".hdf features path", prompt="Path to hdf file"
)
@click.option(
    "--fname",
    default="idf_new",
    help="Name for file to store altered IDFs. If not set will override.",
    prompt="Name for file to store altered IDFs. If not set will override.",
)
@click.option(
    "--log_level",
    default="ERROR",
    help="Logging level",
    prompt="Logging level",
)
def main(hdf, fname, log_level):
    logger.setLevel(log_level)
    set_validation_ventilation_schedules(hdf, fname)


if __name__ == "__main__":
    import pandas as pd

    # mech_vent_sched_mode = 2
    # epjson = EpJsonIDF(
    #     "D:/DATA/validation_v2/idf/000000000.idf",
    # )
    # validation_ventilation(epjson, mech_vent_sched_mode)

    main()
    # hdf_path = "./ml-for-bem/data/temp/validation/v3/features.hdf"
