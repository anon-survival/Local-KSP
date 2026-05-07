
import torch
import torch.nn.functional as F
import util
from util.psr.proper_scoring_rule import DistributionLinear
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_params_to_cat(pred_params, args):
    pred = util.CatDist(pred_params, args)

    return pred

def pred_params_to_weibull(pred_params):
    pre_scale = pred_params[:, 0]
    # scale = pre_scale + 1.
    scale = pre_scale
    pre_k = pred_params[:, 1]
    # k = pre_k.sigmoid() + 1.0
    k = F.softplus(pre_k) + 1e-4
    pred = torch.distributions.Weibull(scale, k)

    return pred

def pred_params_to_lognormal_params(pred_params):
    mu = pred_params[:, 0]
    pre_log_sigma = pred_params[:, 1]
    log_sigma = F.softplus(pre_log_sigma) - 0.5
    sigma = log_sigma.clamp(max=10).exp()
    sigma = sigma + 1e-4

    return mu, sigma

def pred_params_to_lognormal(pred_params):
    mu, sigma = pred_params_to_lognormal_params(pred_params)
    pred = torch.distributions.LogNormal(mu, sigma)

    return pred

def pred_params_to_psr(pred_params, args):
    pred = DistributionLinear(args, args.bin_boundaries)

    return pred

def pred_params_to_dist(pred_params, tgt, args):
    if args.model_dist == 'lognormal':
        pred = pred_params_to_lognormal(pred_params)
    
    elif args.model_dist == 'weibull':
        pred = pred_params_to_weibull(pred_params)

    elif args.model_dist in ['cat', 'mtlr']:
        pred = pred_params_to_cat(pred_params, args)

    elif args.model_dist == 'psr':
        pred = pred_params_to_psr(pred_params, args)

    else:
        pred = pred_params_to_cox(pred_params, tgt)

    return pred

def pred_params_to_cox(pred_params, tgt):
    EPS = 1e-8
    risk_score = torch.exp(pred_params)
    tte, is_dead = tgt[:, 0], tgt[:, 1]

    order = torch.argsort(tte)
    tte = tte[order].reshape(-1)
    is_dead = is_dead[order]
    risk_score = risk_score[order].reshape(-1)

    unique_times, inverse_indices = torch.unique(tte, return_inverse=True)
    n_unique = unique_times.shape[0]

    event_count = torch.zeros(n_unique, device=DEVICE).scatter_add(0, inverse_indices, is_dead)

    # risk_at_time = torch.zeros(n_unique, device=DEVICE)
    # for i in range(n_unique):
    #     risk_at_time[i] = risk_score[tte >= unique_times[i]].sum()
    
    risk_cumsum = torch.flip(torch.cumsum(torch.flip(risk_score, dims=[0]), dim=0), dims=[0])
    first_idx = torch.searchsorted(tte, unique_times)
    risk_at_time = risk_cumsum[first_idx]

    hazard_jump = event_count / (risk_at_time + EPS)
    H = torch.cumsum(hazard_jump, dim=0)

    H_i = H[inverse_indices]
    cdf = 1 - torch.exp(-H_i * risk_score)

    return cdf

def get_cdf_val(pred_params, tgt, args):
    
    pred = pred_params_to_dist(pred_params, tgt, args)

    if args.model_dist in ['cat', 'mtlr']:
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        cdf = pred.cdf(tte, ratio)

    elif args.model_dist == 'psr':
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        cdf = pred.cdf(pred_params, tte, ratio, mask=None)

    elif args.model_dist in ['lognormal', 'weibull']:
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        cdf = pred.cdf(tte + 1e-4)
        
    elif args.model_dist == 'cox':
        cdf = pred
    
    else:
        raise ValueError(f"Unsupported model_dist: {args.model_dist}")

    return cdf
    
def get_cdf_matrix(pred_params, tgt, args):
    pred = pred_params_to_dist(pred_params, tgt, args)
    N = pred_params.shape[0]
    if args.model_dist in ['lognormal', 'weibull']:
        tte, is_dead = tgt[:, 0], tgt[:, 1]
        tte = tte + 1e-4
        tte_expand = tte.unsqueeze(0).expand(N, -1)
        
        pred = pred_params_to_dist(pred_params, tgt=None, args=args)
        
        if args.model_dist == 'lognormal':
            mu, sigma = pred_params_to_lognormal_params(pred_params)
            mu = mu.unsqueeze(1).expand(N, N)
            sigma = sigma.unsqueeze(1).expand(N, N)
            dist = torch.distributions.LogNormal(mu, sigma)
        
        elif args.model_dist == 'weibull':
            scale = pred_params[:, 0].unsqueeze(1).expand(N, N)
            concentration = F.softplus(pred_params[:, 1]) + 1e-4
            dist = torch.distributions.Weibull(scale, concentration)
        
        cdf_matrix = dist.cdf(tte_expand)
    
    elif args.model_dist in ['cat', 'mtlr', 'psr']:
        tte, is_dead, ratio = tgt[:, 0], tgt[:, 1], tgt[:, 2]
        tte_expand = tte.unsqueeze(0).expand(N, -1)
        ratio = ratio.unsqueeze(0).expand(N, -1)

        if args.model_dist == 'psr':
            cdf_matrix = pred.cdf_matrix(pred_params, tte_expand, ratio)

        else:
            cdf_matrix = pred.cdf_matrix(tte_expand, ratio)

    elif args.model_dist == 'cox':
        EPS = 1e-8
        risk_score = torch.exp(pred_params)
        tte, is_dead = tgt[:, 0], tgt[:, 1]

        order = torch.argsort(tte)
        tte = tte[order]
        is_dead = is_dead[order]
        risk_score = risk_score[order]

        unique_times, inverse_indices = torch.unique(tte, return_inverse=True)

        event_count = torch.zeros_like(unique_times).scatter_add(0, inverse_indices, is_dead)

        risk_at_time = torch.zeros_like(unique_times)
        for i, t in enumerate(unique_times):
            risk_at_time[i] = risk_score[tte >= t].sum()

        hazard_jump = event_count / (risk_at_time + EPS)
        H = torch.cumsum(hazard_jump, dim=0)

        H_i = H[inverse_indices]

        cdf_matrix = 1 - torch.exp(-H_i * risk_score)

    else:
        raise ValueError("Unsupported distribution in get_cdf_matrix")

    return cdf_matrix

def get_predict_time(pred, args):
    if args.model_dist in ['cat', 'mtlr']:
        return pred.predict_time()
    
    elif args.model_dist == 'psr':
        return pred.predict_time()
    
    elif args.model_dist == 'lognormal':
        if args.pred_type == 'mean':
            pred_time = pred.mean

        elif args.pred_type == 'mode':
            pred_time = util.log_normal_mode(pred)

    elif args.model_dist == 'weibull':
        # logtwo = torch.tensor([2.0]).to(DEVICE).log()
        # inverse_concentration = 1.0 / pred.concentration
        # pred_time = pred.scale * logtwo.pow(inverse_concentration)
        # pred_time = pred.scale * torch.exp(torch.special.gammaln(1.0 + 1.0 / pred.concentration))
        pred_time = pred.mean

        if torch.any(torch.isnan(pred_time)) or torch.any(torch.isinf(pred_time)):
            print(":(")
    
    else:
        assert False, "wrong dist or pred type in predict time in utils"
    
    return pred_time

def get_logpdf_val(pred_params, tgt, args):

    pred = pred_params_to_dist(pred_params, tgt, args)
    tte = tgt[:, 0]

    if args.model_dist in ['lognormal', 'weibull']:
        tte = tte + 1e-4
    log_prob = pred.log_prob(tte)
    
    return log_prob

def log_normal_mode(pytorch_distribution_object):
    return (pytorch_distribution_object.loc - pytorch_distribution_object.scale.pow(2)).exp()
