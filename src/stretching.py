import numpy as np


def stretching(
    Vstretching,
    theta_s,
    theta_b,
    hc,
    N,
    kgrid=0,
    report=False,
):
    """
    Python translation of Rutgers ROMS stretching.m

    Parameters
    ----------
    Vstretching : int
    theta_s : float
    theta_b : float
    hc : float
    N : int
    kgrid : int
        0 = rho levels
        1 = w levels

    Returns
    -------
    s : ndarray
    C : ndarray
    """

    if Vstretching not in [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Illegal Vstretching={Vstretching}"
        )

    Np = N + 1

    #
    # Define s-coordinate.
    #

    if Vstretching in [1, 2, 3, 4]:

        ds = 1.0 / N

        if kgrid == 1:
            lev = np.arange(N + 1)
            s = (lev - N) * ds
        else:
            lev = np.arange(1, N + 1) - 0.5
            s = (lev - N) * ds

    #
    # Song & Haidvogel (1994)
    #

    if Vstretching == 1:

        if theta_s > 0:

            Ptheta = np.sinh(theta_s * s) / np.sinh(theta_s)

            Rtheta = (
                np.tanh(theta_s * (s + 0.5))
                / (2.0 * np.tanh(0.5 * theta_s))
                - 0.5
            )

            C = (
                (1 - theta_b) * Ptheta +
                theta_b * Rtheta
            )

        else:

            C = s.copy()

    #
    # Shchepetkin (2005)
    #

    elif Vstretching == 2:

        alfa = 1.0
        beta = 1.0

        if theta_s > 0:

            Csur = (
                1.0 - np.cosh(theta_s * s)
            ) / (
                np.cosh(theta_s) - 1.0
            )

            if theta_b > 0:

                Cbot = (
                    -1.0 +
                    np.sinh(
                        theta_b * (s + 1.0)
                    ) / np.sinh(theta_b)
                )

                weight = (
                    (s + 1.0) ** alfa *
                    (
                        1.0 +
                        (alfa / beta) *
                        (1.0 - (s + 1.0) ** beta)
                    )
                )

                C = (
                    weight * Csur +
                    (1.0 - weight) * Cbot
                )

            else:

                C = Csur

        else:

            C = s.copy()

    #
    # Geyer BBL
    #

    elif Vstretching == 3:

        if theta_s > 0:

            exp_s = theta_s
            exp_b = theta_b
            alpha = 3.0

            Cbot = (
                np.log(
                    np.cosh(
                        alpha * (s + 1.0) ** exp_b
                    )
                )
                / np.log(np.cosh(alpha))
                - 1.0
            )

            Csur = (
                -np.log(
                    np.cosh(
                        alpha * np.abs(s) ** exp_s
                    )
                )
                / np.log(np.cosh(alpha))
            )

            weight = (
                1.0 -
                np.tanh(alpha * (s + 0.5))
            ) / 2.0

            C = (
                weight * Cbot +
                (1.0 - weight) * Csur
            )

        else:

            C = s.copy()

    #
    # Shchepetkin (2010)
    #

    elif Vstretching == 4:

        if theta_s > 0:

            Csur = (
                1.0 - np.cosh(theta_s * s)
            ) / (
                np.cosh(theta_s) - 1.0
            )

        else:

            Csur = -s ** 2

        if theta_b > 0:

            C = (
                np.exp(theta_b * Csur) - 1.0
            ) / (
                1.0 - np.exp(-theta_b)
            )

        else:

            C = Csur

    #
    # Souza et al. (2015)
    #

    elif Vstretching == 5:

        if kgrid == 1:

            lev = np.arange(N + 1)

            s = (
                -(lev**2
                  - 2.0 * lev * N
                  + lev
                  + N**2
                  - N)
                / (N**2 - N)
                - 0.01
                * (lev**2 - lev * N)
                / (1.0 - N)
            )

            s[0] = -1.0

        else:

            lev = np.arange(1, N + 1) - 0.5

            s = (
                -(lev**2
                  - 2.0 * lev * N
                  + lev
                  + N**2
                  - N)
                / (N**2 - N)
                - 0.01
                * (lev**2 - lev * N)
                / (1.0 - N)
            )

        if theta_s > 0:

            Csur = (
                1.0 - np.cosh(theta_s * s)
            ) / (
                np.cosh(theta_s) - 1.0
            )

        else:

            Csur = -s**2

        if theta_b > 0:

            C = (
                np.exp(theta_b * Csur) - 1.0
            ) / (
                1.0 - np.exp(-theta_b)
            )

        else:

            C = Csur

    if report:

        print(
            f"Vstretching={Vstretching} "
            f"theta_s={theta_s} "
            f"theta_b={theta_b} "
            f"hc={hc}"
        )

    return s, C