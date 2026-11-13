import numpy as np

def convert_time_to_days(timevar, epoch="1950-01-01"):

    epoch = np.datetime64(epoch)

    if hasattr(timevar, "values"):
        timevar = timevar.values

    return np.asarray((timevar - epoch) / np.timedelta64(1, "D"), dtype=np.float64)