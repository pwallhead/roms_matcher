"""
It is possible to run on the login mode, but this can be very slow.
It is better to run from a slurm script, or (easiest) in an interactive session on a compute node:

 salloc --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G --time=02:00:00 --account=nn8103k

To use this script, first load the needed modules and then add the xarray installation to PATH
and the roms_matcher code to PYTHONPATH:
 module load NRIS/CPU
 export PATH="/cluster/projects/nn8103k/containerized_python_envs/xarray-env//bin:$PATH"
 export PYTHONPATH="${PYTHONPATH}:/cluster/projects/nn8103k/Phils_pyfiles"
Or use the script:
 source load_modules_for_matchups.sh

Then:
 time python3 make_matchups_a20.py

NOTE: We tried first to enable this script using the 'standard' module set from load_olivia_modules.sh,
and installing xarray via a non-containerized virtual environment. This led to a strange error message
related to Open MPI even though all the packages seemed to import fine.
The same problem occurred trying to use the cvenv approach (temporary virtual environment).
We therefore follow the new recommended containerized approach described here:
https://www.sigma2.no/documentation/hpc_machines/olivia/software_stack.html#olivia-software
which has the advantage of creating fewer python files, but does require the new NRIS module set.

NOTE: We have verified that identical results are obtained when reading data from the raw files
(e.g. in a20_v3aa_dia) vs. the concatenated files. The matchup was slightly faster using yearly-
concatenated 1-variable files at daily resolution (for Chl_s) rather than the 28-day raw files, but the
difference was not very significant. We therefore recommend to use the raw files unless there is
already a set of concatenated files that is suitably chunked to avoid large file sizes (e.g. >2 GB per file).

NOTE: The code has been validated against matlab results (fnmatch_to_model.m) for several problems.
Example: Matchup A20 model results for temperature at weekly resolution over 25 years (322*242*35*1300 points),
         from yearly concatenated files, with observational dataset for '40N' (37e6 observations during 1995-2019),
         Run time Matlab R2019b on home PC (Dell Precision 5530):  444 s
         Run time Python 3.6.8  on home PC (Dell Precision 5530): 1249 s
         Run time Python 3.14.7 on OLIVIA compute node (2026):     378 s
         Maximum Absolute Discrepancy matlab-vs-python over 14e6 not-NaN matchups = 9.5e-7 degC.
         NaN assignments identical between the two codes. 
         Flag values identical at not-NaN matchup data.

Phil Wallhead (NIVA) 23/09/2026
"""

from pathlib import Path

import numpy as np
import xarray as xr

from roms_matcher import ROMSMatcher
from a20_projection import A20Projection
from time_utils import convert_time_to_days
from model_utils import load_model_chunk

dir0 = "/cluster/projects/nn8103k/A20/"

GRID_FILE = dir0 + "Grid/A20niva_grd_v4.nc"

#MODEL_DIR = dir0 + "STORAGE/A20_hindcast_1995_2019_v3aa/concatenated/a20_v3aa_ECO_no3_by_year/"
#MODEL_DIR = dir0 + "STORAGE/A20_hindcast_1995_2019_v3aa/concatenated/a20_v3aa_Chl_s_by_year/"
#MODEL_DIR = dir0 + "STORAGE/A20_hindcast_1995_2019_v3aa/a20_v3aa_dia/"
#MODEL_DIR = dir0 + "STORAGE/A20_hindcast_1995_2019_v3aa/concatenated/a20_v3aa_temp_by_year/"
MODEL_DIR = dir0 + "STORAGE/A20_hindcast_1995_2019_v3ae/a20_v3ae_avg/"

MODEL_FILES = sorted(Path(MODEL_DIR).glob("*.nc"))

#OBS_FILE = dir0 + "observations/ARSEC_obs_data/NO3NO2_obs_1980_2020_40N.nc"
#OBS_FILE = dir0 + "observations/ARSEC_obs_data/Chl_obs_1980_2020_40N.nc"
OBS_FILE = dir0 + "observations/ARSEC_obs_data/T_obs_1980_2020_40N.nc"

#OUTFILE = dir0 + "ARSEC_data_model/A20_v3aa/NO3NO2_obs_a20_v3aa_1980_2020_40N.nc"
#OUTFILE = dir0 + "ARSEC_data_model/A20_v3aa/Chl_obs_a20_v3aa_1980_2020_40N.nc"
#OUTFILE = dir0 + "ARSEC_data_model/A20_v3aa/Chl_obs_a20_v3aa_1980_2020_40N_2.nc"
#OUTFILE = dir0 + "ARSEC_data_model/A20_v3aa/T_obs_a20_v3aa_1980_2020_40N.nc"
OUTFILE = dir0 + "ARSEC_data_model/A20_v3ae/T_obs_a20_v3ae_1980_2020_40N.nc"

# ROMS model vertical grid parameters (here manual input - check consistent with roms.in)
vgrid = {
    "Vtransform": 2,
    "Vstretching": 2,
    "theta_s": 6.0,
    "theta_b": 0.1,
    "hc": 100.0,
    "N": 35,
}

# Model output variable name
varnamem = "temp"

# Factor to convert model output to observational units 
#facm = 1 / (6.625 * 12.01) #For ECO_no3: convert [mgC/m3] to [mmolN/m3] using factors 6.625 and 12.01 from ecosmo_operational.F90.
facm = 1 #For (Chl_s,T) no units conversion.

# Model time variable name
timevarm = "ocean_time"

####### USERS SHOULD NOT NEED TO MODIFY BELOW THIS LINE ########

# Small time tolerance (days) to ensure that observations are not included
# beyond the range of model times extracted. This guards against tiny
# discrepancies arising from differences in numerical representation of times.
# time_tol must remain much smaller than the temporal resolution of the
# model output.
time_tol = 1e-6

#
# Read observations
#
obs = xr.open_dataset(OBS_FILE)
obs_time = convert_time_to_days(obs["time"], epoch="1950-01-01")
nobs = len(obs_time)
date0 = np.datetime64("1950-01-01")

model_value = np.full(nobs, np.nan)
flag = np.zeros(nobs, dtype=np.int8)

#
# Build matcher once
#
matcher = ROMSMatcher(
    grid_file=GRID_FILE,
    vgrid=vgrid,
    coordinate_transform=A20Projection(),
    allow_surface_extrapolation=True,
    allow_bottom_extrapolation=True,
)

#
# Loop over model files
#
for ifile in range(len(MODEL_FILES)):

    print(f"Processing file {MODEL_FILES[ifile].name}, "f"File no. {ifile+1}/{len(MODEL_FILES)}")

    ds_chunk, i0, i1 = load_model_chunk(MODEL_FILES, ifile, time_var=timevarm)

    #
    # Attach chunk to matcher
    #
    matcher.ds = ds_chunk
    matcher.time = convert_time_to_days(ds_chunk[timevarm], epoch="1950-01-01")

    #
    # Time range corresponding to the present chunk
    #
    tmin = matcher.time[0]
    tmax = matcher.time[-1]

    #
    # Select observations belonging to this file
    #
    sel = ((obs_time >= tmin+time_tol) & (obs_time <= tmax-time_tol))
    # NOTE:
    # Observations in the overlap between adjacent files may be processed more than once.
    # This is intentional. The interpolation stencil is identical, so the result simply overwrites
    # itself and ensures complete temporal coverage. The small shifts by time_tol ensure that
    # we do not include observations beyond the extracted interpolation range of the model
    # (see comment above).

    obs_ids = np.where(sel)[0]

    if len(obs_ids) == 0:
        continue

    tmin_date = date0 + np.timedelta64(int(round(tmin)), "D")
    tmax_date = date0 + np.timedelta64(int(round(tmax)), "D")
    print(f"Found {len(obs_ids)} observations for this chunk "f"({tmin_date} to {tmax_date})")

    #
    # Matchups for this chunk
    #
    value_chunk, flag_chunk = (matcher.match_variable(
            variable=varnamem,
            obs_lon=obs["lon"].values[obs_ids],
            obs_lat=obs["lat"].values[obs_ids],
            obs_depth=obs["depth"].values[obs_ids],
            obs_time=obs_time[obs_ids]))

    #
    # Store results
    #
    model_value[obs_ids] = value_chunk
    flag[obs_ids] = flag_chunk

    #
    # Explicitly close files
    #
    ds_chunk.close()

#
# Write output
#
out = obs.copy()
out[varnamem] = (("datum",), model_value * facm)
out["flag"] = (("datum",), flag)
out.to_netcdf(OUTFILE)
