import numpy as np

from stretching import stretching


def set_depth(
    Vtransform,
    Vstretching,
    theta_s,
    theta_b,
    hc,
    N,
    igrid,
    h,
    zeta=None,
    report=False,
):
    """
    Python translation of Rutgers ROMS set_depth.m.

    Parameters
    ----------
    Vtransform : int
    Vstretching : int
    theta_s : float
    theta_b : float
    hc : float
    N : int
    igrid : int
        1=rho, 2=psi, 3=u, 4=v, 5=w
    h : ndarray
        Bathymetry (positive)
    zeta : ndarray, optional
        Free surface
    report : bool

    Returns
    -------
    z : ndarray
        Depths (negative)
    """

    if Vtransform not in [1, 2]:
        raise ValueError(
            f"Illegal Vtransform={Vtransform}"
        )

    if Vstretching not in [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Illegal Vstretching={Vstretching}"
        )

    h = np.asarray(h)

    if Vtransform == 1 and hc > np.min(h):
        raise ValueError(
            "hc exceeds minimum bathymetry"
        )

    if zeta is None:
        zeta = np.zeros_like(h)

    Np = N + 1

    if igrid == 5:
        kgrid = 1
    else:
        kgrid = 0

    s, C = stretching(
        Vstretching,
        theta_s,
        theta_b,
        hc,
        N,
        kgrid,
        report=report,
    )

    Lp, Mp = h.shape
    L = Lp - 1
    M = Mp - 1

    #
    # Average h and zeta onto requested grid.
    #

    if igrid == 1:

        hgrid = h
        zetagrid = zeta

    elif igrid == 2:

        hgrid = 0.25 * (
            h[0:L, 0:M] +
            h[1:Lp, 0:M] +
            h[0:L, 1:Mp] +
            h[1:Lp, 1:Mp]
        )

        zetagrid = 0.25 * (
            zeta[0:L, 0:M] +
            zeta[1:Lp, 0:M] +
            zeta[0:L, 1:Mp] +
            zeta[1:Lp, 1:Mp]
        )

    elif igrid == 3:

        hgrid = 0.5 * (
            h[0:L, :] +
            h[1:Lp, :]
        )

        zetagrid = 0.5 * (
            zeta[0:L, :] +
            zeta[1:Lp, :]
        )

    elif igrid == 4:

        hgrid = 0.5 * (
            h[:, 0:M] +
            h[:, 1:Mp]
        )

        zetagrid = 0.5 * (
            zeta[:, 0:M] +
            zeta[:, 1:Mp]
        )

    elif igrid == 5:

        hgrid = h
        zetagrid = zeta

    else:

        raise ValueError(
            f"Illegal igrid={igrid}"
        )

    #
    # Allocate output.
    #

    if igrid == 5:
        nz = N + 1
    else:
        nz = N

    z = np.empty(
        (*hgrid.shape, nz),
        dtype=np.float64,
    )

    #
    # Vtransform = 1
    #

    if Vtransform == 1:

        if igrid == 5:

            z[:, :, 0] = -hgrid

            for k in range(1, Np):

                z0 = (
                    (s[k] - C[k]) * hc +
                    C[k] * hgrid
                )

                z[:, :, k] = (
                    z0 +
                    zetagrid * (1.0 + z0 / hgrid)
                )

        else:

            for k in range(N):

                z0 = (
                    (s[k] - C[k]) * hc +
                    C[k] * hgrid
                )

                z[:, :, k] = (
                    z0 +
                    zetagrid * (1.0 + z0 / hgrid)
                )

    #
    # Vtransform = 2
    #

    elif Vtransform == 2:

        for k in range(nz):

            z0 = (
                hc * s[k] +
                C[k] * hgrid
            ) / (hc + hgrid)

            z[:, :, k] = (
                zetagrid +
                (zetagrid + hgrid) * z0
            )

    return z