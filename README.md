This code enables efficient 4D interpolation of ROMS model output to an arbitrary set of observed locations/times.
Coastward extrapolation is strictly forbidden; vertical extrapolation (above or below) is allowed but flagged (flag value 1 or 2).

The algorithm proceeds by identifying a set of model grid point squares (from 4 rho points) that enclose observed locations.  
Interpolated values for any observations with a land mask point in the surrounding square will be returned as NaN.
For each sampled square we loop over enclosed observations, and for each observation: 
1) interpolate in 2D (t,z) to the observed (time, depth) for each of the 4 enclosing model water columns,
2) finally interpolate in 2D (x,y) the values from step (1) to the observed horizontal location.

A driver script is needed to use the code (see folder example_driver_scripts for examples).
Preprocessing of model output (e.g. extracting single variables and concatenating) may improve performance but is not strictly needed.
The script can operate on a folder of raw ROMS output files (e.g. 'avg' files), and will proceed by looping over observations within the
time span of each file, appended by the last time stamp of the previous file and the first of the following file where available.

Optionally, the user may provide a custom projection to convert observed (latitude,longitude) to (x,y) in model coordinates (see example_driver_scripts/a20_projection.py).
Otherwise, (x,y) coordinates will be calculated by interpolating values from the model grid file.

The algorithm does not account for time dependence in the model depth values.
This should be a good enough approximation except in cases of very shallow water columns with very strong tides.
Optionally, the user may provide a fixed, time-independent set of surface elevations 'zeta_mean' to refine the depth calculations, otherwise zeta = 0 will be assumed.
