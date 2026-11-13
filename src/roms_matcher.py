import numpy as np
import xarray as xr

from scipy.interpolate import LinearNDInterpolator
from set_depth import set_depth


class ROMSMatcher:

    def __init__(
        self,
        grid_file,
        vgrid,
        coordinate_transform=None,
        zeta_mean=None,
        allow_surface_extrapolation=True,
        allow_bottom_extrapolation=True,
    ):

        self.Vtransform = vgrid["Vtransform"]
        self.Vstretching = vgrid["Vstretching"]
        self.theta_s = vgrid["theta_s"]
        self.theta_b = vgrid["theta_b"]
        self.hc = vgrid["hc"]
        self.N = vgrid["N"]

        self.zm_deep_to_shallow = True #ROMS data are always indexed in order deep-to-shallow

        self.coordinate_transform = coordinate_transform

        self.allow_surface_extrapolation = allow_surface_extrapolation
        self.allow_bottom_extrapolation = allow_bottom_extrapolation

        self.grid = xr.open_dataset(grid_file)

        self.mask_rho = self.grid["mask_rho"].values.astype(np.int8)
        self.h = self.grid["h"].values.astype(np.float64)

        self.lat_rho = self.grid["lat_rho"].values.astype(np.float64)
        self.lon_rho = self.grid["lon_rho"].values.astype(np.float64)

        if coordinate_transform is not None:
            # Recompute projected coordinates for grid points
            X, Y = coordinate_transform(self.lon_rho, self.lat_rho)

            tol = 1e-4
            if np.max(np.abs(X - X[[0], :])) > tol:
                raise ValueError("Transformed X coordinates are not separable")
            if np.max(np.abs(Y - Y[:, [0]])) > tol:
                raise ValueError("Transformed Y coordinates are not separable")

            self.xm = X[0, :]
            self.ym = Y[:, 0]
        else:
            nx, ny = self.lat_rho.shape
            self.xm = np.arange(nx)
            self.ym = np.arange(ny)

        self.build_lonlat_mapper()
        self.build_vertical_grid(zeta_mean)

    def build_lonlat_mapper(self):

        X, Y = np.meshgrid(self.ym, self.xm)

        self.fx = LinearNDInterpolator(
            np.column_stack([
                self.lat_rho.ravel(),
                self.lon_rho.ravel()
            ]),
            X.ravel()
        )

        self.fy = LinearNDInterpolator(
            np.column_stack([
                self.lat_rho.ravel(),
                self.lon_rho.ravel()
            ]),
            Y.ravel()
        )

    def lonlat_to_xy(self, lon, lat):

        if self.coordinate_transform is not None:
            return self.coordinate_transform(lon, lat)

        x = self.fx(lat, lon)
        y = self.fy(lat, lon)

        return x, y

    def build_vertical_grid(self, zeta_mean=None):

        if zeta_mean is None:
            zeta_mean = np.zeros_like(self.h)

        z = set_depth(self.Vtransform, 
                      self.Vstretching, 
                      self.theta_s, 
                      self.theta_b, 
                      self.hc, 
                      self.N, 
                      1,                 # rho points
                      self.h, 
                      zeta_mean,
                      report=False)

        Zm = -z + zeta_mean[:, :, None]  # depths should be positive downwards and relative to sea surface, to compare with observational depth.
        if self.zm_deep_to_shallow:
            Zm = Zm[:, :, ::-1]          # depths should be ordered small-to-large (shallow-to-deep) for use in match_variable

        Zm = np.where(self.mask_rho[:, :, None] == 0, np.nan, Zm)

        self.Zm = Zm

    def group_observations(self, xobs, yobs):

        dxm = np.diff(self.xm)
        dym = np.diff(self.ym)

        tol = 1e-4

        if (np.max(np.abs(dxm-dxm[0])) > tol or np.max(np.abs(dym-dym[0])) > tol):
            raise ValueError("Grid not regular in (x,y)")

        dx = dxm[0]
        dy = dym[0]

        xmin = self.xm.min()
        xmax = self.xm.max()

        ymin = self.ym.min()
        ymax = self.ym.max()

        ixobs = np.floor((xobs-xmin)/dx).astype(int)
        iyobs = np.floor((yobs-ymin)/dy).astype(int)

        good = ((xobs > xmin) & (xobs < xmax) & (yobs > ymin) & (yobs < ymax))
        selgood = np.where(good)[0]

        Cixy, ICixy = np.unique(np.column_stack([ixobs[selgood], iyobs[selgood]]), axis=0, return_inverse=True)

        obs_in_cell = [[] for _ in range(len(Cixy))]

        for obs_idx, icell in zip(selgood, ICixy):
            obs_in_cell[icell].append(obs_idx)

        return (Cixy, obs_in_cell, ixobs, iyobs)

    def match_variable(
        self,
        variable,
        obs_lon,
        obs_lat,
        obs_depth,
        obs_time,
    ):

        nobs = len(obs_lon)

        values = np.full(nobs, np.nan, dtype=np.float32)
        flags = np.zeros(nobs, dtype=np.int8)

        xobs, yobs = self.lonlat_to_xy(obs_lon, obs_lat)

        cells, obs_in_cell, ixobs, iyobs = self.group_observations(xobs, yobs)

        V = self.ds[variable].values
        if self.zm_deep_to_shallow:
            V = V[:, ::-1, :, :]

        for icell, (ix, iy) in enumerate(cells):

            if (self.mask_rho[iy:iy+2, ix:ix+2] == 0).any():
                continue

            obs_ids = np.asarray(obs_in_cell[icell], dtype=int)

            if len(obs_ids) == 0:
                continue

            x0 = self.xm[ix]
            x1 = self.xm[ix+1]

            y0 = self.ym[iy]
            y1 = self.ym[iy+1]

            wx = (xobs[obs_ids]-x0)/(x1-x0)
            wy = (yobs[obs_ids]-y0)/(y1-y0)

            w11 = (1-wx)*(1-wy)
            w12 = (1-wx)*wy
            w21 = wx*(1-wy)
            w22 = wx*wy

            c11 = np.full(len(obs_ids), np.nan)
            c12 = np.full(len(obs_ids), np.nan)
            c21 = np.full(len(obs_ids), np.nan)
            c22 = np.full(len(obs_ids), np.nan)

            for n, obs_idx in enumerate(obs_ids):

                t_obs = obs_time[obs_idx]
                z_obs = obs_depth[obs_idx]

                it = np.searchsorted(self.time, t_obs)
                if it == 0 or it >= len(self.time):
                    continue
                it0 = it-1
                it1 = it

                t0 = self.time[it0]
                t1 = self.time[it1]

                ft = (t_obs - t0)/(t1 - t0)

                # Interpolate bilinearly over (z,t) at each of the 4 surrounding model points
                c11[n], f11 = self.interpolate_time_depth(V, ix, iy,     z_obs, it0, it1, ft)
                c12[n], f12 = self.interpolate_time_depth(V, ix, iy+1,   z_obs, it0, it1, ft)
                c21[n], f21 = self.interpolate_time_depth(V, ix+1, iy,   z_obs, it0, it1, ft)
                c22[n], f22 = self.interpolate_time_depth(V, ix+1, iy+1, z_obs, it0, it1, ft)

                flags[obs_idx] = max(f11, f12, f21, f22)

            good = (~np.isnan(c11) & ~np.isnan(c12) & ~np.isnan(c21) & ~np.isnan(c22))

            # Interpolate bilinearly over (x,y)
            values_cell = (w11*c11 + w12*c12 + w21*c21 + w22*c22)

            values[obs_ids[good]] = values_cell[good]

        return values, flags

    def interpolate_time_depth(self,V,i,j,z_obs,it0,it1,ft):

        zcol = self.Zm[j, i, :]

        flag = 0
        zqc = z_obs

        if z_obs < zcol[0]:

            flag = 1
            if not self.allow_surface_extrapolation:
                return np.nan, flag
            zqc = zcol[0]

        elif z_obs > zcol[-1]:

            flag = 2
            if not self.allow_bottom_extrapolation:
                return np.nan, flag
            zqc = zcol[-1]

        iz = np.searchsorted(zcol, zqc)

        iz0 = iz-1
        iz1 = iz

        z0 = zcol[iz0]
        z1 = zcol[iz1]

        v00 = V[it0, iz0, j, i]
        v01 = V[it0, iz1, j, i]
        v10 = V[it1, iz0, j, i]
        v11 = V[it1, iz1, j, i]

        if np.isnan([v00, v01, v10, v11]).any():
            return np.nan, flag

        if z1 == z0:
            fz = 0.0
        else:
            fz = (zqc-z0)/(z1-z0)

        vt0 = (1-fz)*v00 + fz*v01
        vt1 = (1-fz)*v10 + fz*v11

        value = (1-ft)*vt0 + ft*vt1

        return value, flag