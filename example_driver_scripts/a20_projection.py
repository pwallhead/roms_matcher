import numpy as np


class A20Projection:
    """
    Exact translation of

        GridData = [209 128.5 20 58 60 1 6371]
        latlon2xy.m

    Returns coordinates in A20 grid units
    (same units as xi_rho/20000 and eta_rho/20000).
    """

    def __init__(self):

        self.GridData = np.array(
            [209, 128.5, 20, 58, 60, 1, 6371],
            dtype=float
        )

    def __call__(self, lon, lat):

        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)

        RAD = np.pi / 180.0

        xpole = self.GridData[0]
        ypole = self.GridData[1]

        dx = self.GridData[2]

        angle = self.GridData[3]
        lat0 = self.GridData[4]

        Re = self.GridData[6]

        AN = Re * (1.0 + np.sin(lat0 * RAD)) / dx

        VXR = angle * RAD

        PLR = lon * RAD
        PBR = lat * RAD

        R = AN * np.cos(PBR) / (1.0 + np.sin(PBR))

        X = (
            R
            * (
                np.sin(PLR) * np.cos(VXR)
                - np.sin(VXR) * np.cos(PLR)
            )
            + xpole
        )

        Y = (
            -R
            * (
                np.cos(PLR) * np.cos(VXR)
                + np.sin(PLR) * np.sin(VXR)
            )
            + ypole
        )

        return X, Y