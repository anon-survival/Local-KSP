import torch

import models
import optim
import util
from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import time


def compute_scal(pred_params, tgt, args):
    cdf = util.get_cdf_val(pred_params, tgt, args)

    tte, is_dead = tgt[:, 0], tgt[:, 1]
    if args.model_dist == 'cox':
        order = torch.argsort(tte)
        is_dead = is_dead[order]

    s_cal = util.s_calibration(points=cdf, phase='train', is_dead=is_dead, args=args, gamma=args.train_gamma, differentiable=True, device=DEVICE)

    return s_cal

def train(args):
    train_loader = util.get_train_loader(args)

    args.device = DEVICE
    
    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args)
        args.start_epoch = ckpt_info['epoch'] + 1

    else:
        model_fn = models.__dict__[args.model]
        args.D_in = train_loader.D_in
        model = model_fn(**vars(args))

    model = model.to(args.device)
    
    model.train()
    optimizer = optim.get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    lr_scheduler = optim.get_scheduler(optimizer, args)
    
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
    
    loss_fn = optim.get_loss_fn(args.loss_fn, args)

    logger = TrainLogger(args, len(train_loader.dataset))

    eval_loaders = util.get_eval_loaders(during_training=True, args=args)

    evaluator = ModelEvaluator(args, eval_loaders)
    
    saver = ModelSaver(**vars(args))

    with torch.no_grad():
        metrics = evaluator.evaluate(model, args.device, 0)

    if args.lam > 0.0:
        lam = args.lam

    else:
        lam = 0.0
    
    time_cum = []
    while not logger.is_finished_training():
        logger.start_epoch()
        s_cal_accumulator = 0.0

        print("******* STARTING TRAINING LOOP *******")
        start = time.time()
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        
        print('current lam', lam)
        print('current lr', cur_lr)

        for src, tgt in train_loader:
            logger.start_iter()

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            with torch.set_grad_enabled(True):
                if torch.any(torch.isnan(src)):
                    print("SRC HAS NAN")

                if torch.any(torch.isnan(tgt)):
                    print("TGT HAS NAN")
                
                model.train()
                pred_params = model.forward(src.to(args.device))
                
                if args.model_dist in ['cat', 'mtlr']:
                    tgt = util.cat_bin_target(args, tgt, bin_boundaries)
                    #weight = model.get_weight()

                loss = 0
                if not args.loss_scal_only:
                    """
                    if args.model_dist in ['mtlr']:
                        loss += loss_fn(pred_params, tgt, model_dist=args.model_dist) + util.ridge_norm(weight)*args.C1/2 + util.fused_norm(weight)*args.C2/2
                    else:
                        loss += loss_fn(pred_params, tgt, model_dist=args.model_dist)
                    """
                    loss += loss_fn(pred_params, tgt, model_dist=args.model_dist)
                if args.lam > 0 or args.loss_scal_only:
                    s_cal = compute_scal(pred_params, tgt, args)

                    s_cal_accumulator += s_cal.detach().item()

                    if args.loss_scal_only:
                        loss = s_cal

                    else:
                        loss = loss + lam * s_cal

                logger.log_iter(src, pred_params, tgt, loss)
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            logger.end_iter()
        end = time.time()
        print(f"{end - start:.5f} sec")
        time_cum.append([end - start])
        print(np.mean(time_cum), np.std(time_cum))
        print("********** CALLING EVAL **********")

        with torch.no_grad():
            metrics = evaluator.evaluate(model, args.device, logger.epoch)

        saver.save(logger.epoch, model, optimizer, lr_scheduler, args.device,\
                   metric_val=metrics.get(args.metric_name, None))
        logger.end_epoch(metrics=metrics)

        if args.lr_scheduler != 'none':
            optim.step_scheduler(lr_scheduler, metrics, logger.epoch)
            print("ATTEMPT STEPPING LEARNING RATE")

if __name__ == '__main__':
    torch.set_anomaly_enabled(True)
    parser = TrainArgParser()
    args = parser.parse_args()

    print("CUDA IS AVAILABLE:", torch.cuda.is_available())
    if args.model_dist in ['cat', 'mtlr']:
        bin_boundaries, mid_points = util.get_bin_boundaries(args)
        print('bin_boundaries', bin_boundaries)
        args.bin_boundaries = bin_boundaries
        args.mid_points = mid_points

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print('dataset name', args.dataset)
    train(args)
    