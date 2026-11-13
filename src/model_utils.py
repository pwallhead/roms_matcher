import xarray as xr


def load_model_chunk(files, ifile, time_var):

    """
    Load a model chunk for interpolation.

    Includes:
        - last timestep from previous file (if available)
        - all timesteps from current file
        - first timestep from following file (if available)

    Parameters
    ----------
    files : list
        Ordered list of NetCDF files.

    ifile : int
        Index of current file.

    time_var : str
        Name of time variable.

    Returns
    -------
    ds_chunk : xarray.Dataset

    i0, i1 : int
        Indices within ds_chunk corresponding to
        the current file only.
    """

    dslist = []
    i0 = 0

    #
    # Previous file
    #
    if ifile > 0:
        dsp = xr.open_dataset(files[ifile-1])
        last_time = dsp[time_var].values[-1]
        dsp = dsp.where(dsp[time_var] == last_time, drop=True)
        dslist.append(dsp)
        i0 = 1

    #
    # Current file
    #
    dsc = xr.open_dataset(files[ifile])
    dslist.append(dsc)
    ncurr = dsc.sizes[time_var]

    #
    # Following file
    #
    if ifile < len(files)-1:
        dsn = xr.open_dataset(files[ifile+1])
        first_time = dsn[time_var].values[0]
        dsn = dsn.where(dsn[time_var] == first_time, drop=True)
        dslist.append(dsn)

    ds_chunk = xr.concat(dslist, dim=time_var)

    i1 = i0 + ncurr - 1

    return ds_chunk, i0, i1




def get_chunk_timerange(ds_chunk, i0, i1, time_var):
    """
    Return the time range corresponding to the
    current file (excluding neighbouring timesteps).

    Parameters
    ----------
    ds_chunk : xarray.Dataset

    i0 : int
        First index belonging to current file
        within ds_chunk.

    i1 : int
        Last index belonging to current file
        within ds_chunk.

    time_var : str
        Name of time variable.

    Returns
    -------
    tmin, tmax
    """

    time = ds_chunk[time_var].values

    tmin = time[i0]
    tmax = time[i1]

    return tmin, tmax