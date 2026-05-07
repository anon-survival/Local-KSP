import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_params_to_cox(pred_params, tgt, i):
    EPS = 1e-13
    risk_score = torch.exp(pred_params)
    tte, is_dead = tgt[:, 0], tgt[:, 1]

    order = torch.argsort(tte)
    tte = tte[order].reshape(-1, 1)
    is_dead = is_dead[order]

    tte_diff = tte - tte.T
    mask = (tte_diff <= 0).float().to(DEVICE)
    
    risk_score2 = risk_score[order].reshape(-1)
    risk_value = torch.sum(mask * risk_score2, dim=1) + EPS
    value = is_dead / risk_value
    
    H = torch.cumsum(value, dim=0) # Breslow estimator
    S = torch.exp(-H) + EPS # Baseline survival function
    
    cdf = 1 - S ** risk_score[i]

    return cdf