import torch
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistributionLinear:
    '''
    This class represents probability distribution.
    Key points of cumulative distribution functions are stored as boundaries,
    and they are connected by using linear interpolation.
    '''

    def __init__(self, args, boundaries, axis='target'):
        '''
        axis should be 'target' or 'quantile'
        '''
        self.args = args
        self.boundaries = torch.as_tensor(boundaries, dtype=torch.float32).to(DEVICE)
        if axis=='target':
            self.axis_is_target = True
        elif axis=='quantile':
            self.axis_is_target = False
        else:
            raise ValueError('Unknown axis value'+axis)

    def _interpolate(self, pred, y, ratio, mask):
        pred = pred.to(DEVICE)
        y = y.long().to(DEVICE).view(-1, 1)
        if mask is not None:
            pred = pred[mask]
            y = y[mask]
            ratio = ratio[mask]

        cum_pred = torch.cumsum(pred, dim=1)
        F_pred = torch.cat([torch.zeros(pred.shape[0], 1).to(DEVICE), cum_pred, torch.ones(pred.shape[0], 1).to(DEVICE)], 1)
        # compute idx and ratio
        # idx = torch.searchsorted(self.boundaries, y, right=True).to(DEVICE)
        idx = y
        # idx[idx == 0] = 1
        
        # b_lb = self.boundaries[idx-1]

        # mask_of = (idx < len(self.boundaries))
        # ratio = torch.zeros_like(b_lb).to(DEVICE)
        # b_ub = self.boundaries[idx[mask_of]]
        # ratio[mask_of] = (y[mask_of]-b_lb[mask_of]) / (b_ub-b_lb[mask_of])
        # idx[~mask_of] -= 1
        # ratio[~mask_of] = 1.0
        # print(idx)
        # idx = idx.view(-1, 1)
        # ratio = ratio.view(-1, 1)

        # right = torch.gather(F_pred, 1, idx).view(-1)
        # left = torch.zeros_like(right)
        # mask_zero = (idx > 0).view(-1)
        # left[mask_zero] = torch.gather(F_pred[mask_zero], 1, (idx[mask_zero] - 1)).view(-1)

        left = torch.gather(F_pred, 1, idx).view(-1)
        right = torch.gather(F_pred, 1, idx+1).view(-1)

        return torch.lerp(left, right, ratio.float())

    def _interpolate_inv(self, pred, quantiles, ratio, mask):
        if mask is not None:
            pred = pred[mask]
            quantiles = quantiles[mask]
        cum_pred = torch.cumsum(pred, dim=1)
        F_pred = torch.cat([torch.zeros(pred.shape[0], 1).to(DEVICE), cum_pred], 1)

        # compute idx and ratio
        # idx = torch.searchsorted(F_pred, quantiles, right=True)
        idx = quantiles
        # Fs_lb = torch.gather(F_pred, 1, idx-1)
        mask_of = (idx < len(self.boundaries)).view(-1)
        Fs_ub_mask = torch.gather(F_pred[mask_of], 1, idx[mask_of])
        # ratio = torch.zeros_like(Fs_lb)
        # ratio_numerator = quantiles[mask_of] - Fs_lb[mask_of]
        # ratio[mask_of] =  ratio_numerator / (Fs_ub_mask - Fs_lb[mask_of])
        idx[~mask_of] -= 1
        ratio[~mask_of] = 1.0

        # linear interpolation
        left = self.boundaries[idx-1]
        right = self.boundaries[idx]
        return torch.lerp(left, right, ratio)

    def cdf(self, pred, y, ratio, mask=None):
        '''
        Cumulative distribution function.

        Parameters
        ----------
        pred : Tensor
            Each row represents a probability distribution.
            The sum of each row must be equal to one.
            Tensor shape is [batch size, n_bin+1].
        y : Tensor
            Compute CDF of y
            Tensor shape is [batch size, col_size].
        mask : Tensor
            Mask rows of pred and y.
            Tensor shape is [batch size].

        Returns
        -------
        quantiles : Tensor
            Computed quantiles of y.
            Tensor shape is equal to the shape of y.
        '''
        if self.axis_is_target:
            return self._interpolate(pred, y, ratio, mask)
        else:
            return self._interpolate_inv(pred, y, ratio, mask)

    def icdf(self, pred, quantile, mask=None):
        '''
        Inverse of cumulative distribution function.

        Parameters
        ----------
        pred : Tensor
            Piecewise-linear CDF with n_bin+1 endpoints.
            Each row corresponds to a CDF.
            pred[:,0] = 0.0 and pred[:,-1] = 1.0
            Tensor shape is [batch size, n_bin+1].
        quantile : Tensor
            Quantiles
            Tensor shape is [batch size, col_size].
        mask : Tensor
            Mask rows of pred and y.
            Tensor shape is [batch size].

        Returns
        -------
        y : Tensor
            Compute y.
            Tensor shape is equal to the shape of quantile.
        '''
        if self.axis_is_target:
            return self._interpolate_inv(pred, quantile, mask)
        else:
            return self._interpolate(pred, quantile, mask)
        
    def predict_time(self, pred):
        pred_time = util.get_mean_bins2(pred, self.args.mid_points)

        return pred_time
    
    def cdf_matrix(self, pred, times, ratio, chunk_size=2000):
        params = pred  # [n, K]
        n, K = params.shape
        m = times.shape[1]
        # probs = torch.softmax(params, dim=-1).to(torch.float32)  # [n, K]
        probs = params

        chunks = []
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            times_chunk = times[:, start:end]  # [n, chunk]
            ratio_chunk = ratio if isinstance(ratio, float) or ratio.dim() == 0 else ratio[:, start:end]

            times_exp = times_chunk.unsqueeze(-1)            # [n, chunk, 1]
            indices = torch.arange(K, device=times.device).view(1, 1, -1)  # [1, 1, K]
            mask1 = (times_exp > indices).float()            # [n, chunk, K]
            mask2 = (times_exp >= indices).float()           # [n, chunk, K]

            probs_exp = probs.unsqueeze(1).expand(-1, times_chunk.size(1), -1)  # [n, chunk, K]

            cdf_km1 = (probs_exp * mask1).sum(dim=-1)  # [n, chunk]
            prob_k = (probs_exp * (mask2 - mask1)).sum(dim=-1)  # just prob at k
            result_chunk = cdf_km1 + prob_k * (ratio_chunk if not isinstance(ratio, float) else ratio)
            result_chunk = torch.clamp(result_chunk, 1e-6, 1 - 1e-6)

            chunks.append(result_chunk)

        return torch.cat(chunks, dim=1)  # [n, m]