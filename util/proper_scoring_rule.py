import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from scipy.interpolate import PchipInterpolator

def pchip_surv_interp(Y, F_Y_given_z, t_grid, extrapolate=False):
    Y = np.asarray(Y, dtype=float)
    F_Y_given_z = np.asarray(F_Y_given_z, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if F_Y_given_z.shape[1] != len(Y):
        raise ValueError("F_Y_given_z.shape[1] must equal len(Y).")

    unique_Y, inverse = np.unique(Y, return_inverse=True)
    n_rows = F_Y_given_z.shape[0]
    F_unique = np.zeros((n_rows, len(unique_Y)), dtype=float)

    for k in range(len(unique_Y)):
        cols = (inverse == k)
        F_unique[:, k] = F_Y_given_z[:, cols].mean(axis=1)

    F_interp = np.zeros((n_rows, len(t_grid)), dtype=float)

    for i in range(n_rows):
        interpolator = PchipInterpolator(unique_Y, F_unique[i], extrapolate=extrapolate)
        F_interp[i] = interpolator(t_grid)

    F_interp = np.clip(F_interp, 0.0, 1.0)

    return F_interp

def cen_log_simple(tte, is_dead, cdf_matrix, nbins=32, eps=1e-8):
    device = tte.device
    N = tte.shape[0]

    t_grid = torch.linspace(0.0, tte.max(), nbins, device=device)
    cdf_np = pchip_surv_interp(tte.cpu(), cdf_matrix.cpu(), t_grid.cpu())
    cdf_np[np.isnan(cdf_np)] = 0
    cdf = torch.tensor(cdf_np, device=device)

    diff_cdf = torch.diff(cdf, dim=1)
    diff_cdf = torch.clamp(diff_cdf, min=eps)

    bin_idx = torch.bucketize(tte, t_grid) - 1
    row_idx = torch.arange(N, device=device)

    uncensored_part = torch.log(diff_cdf[row_idx, bin_idx])
    censored_part = torch.log(torch.clamp(1 - cdf[row_idx, bin_idx+1], min=eps))

    cen_log = is_dead * uncensored_part + (1 - is_dead) * censored_part

    return -cen_log.mean()