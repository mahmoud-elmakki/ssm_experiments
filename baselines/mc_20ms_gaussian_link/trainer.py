import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from hydra import compose, initialize

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import pytorch_lightning as lightning

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import xfads.utils as utils
import xfads.prob_utils as prob_utils

from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM, LightningMonkeyReaching
from xfads.smoothers.nonlinear_smoother_causal import NonlinearFilter, LowRankNonlinearStateSpaceModel



def main():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    """config"""
    initialize(version_base=None, config_path="", job_name="monkey_reaching")
    cfg = compose(config_name="config")

    lightning.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    """load the data"""
    data_path = '/home/makki/data/data_{split}_{bin_size_ms}ms.pt'
    train_data = torch.load(data_path.format(split='train', bin_size_ms=cfg.bin_sz_ms))
    val_data = torch.load(data_path.format(split='valid', bin_size_ms=cfg.bin_sz_ms))
    test_data = torch.load(data_path.format(split='test', bin_size_ms=cfg.bin_sz_ms))

    y_train_obs = train_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :cfg.n_bins_enc, :]
    y_valid_obs = val_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :cfg.n_bins_enc, :]
    y_test_obs = test_data['y_obs'].type(torch.float32).to(cfg.data_device)[:, :cfg.n_bins_enc, :]

    """Gaussian-smoothed spike trains"""
    y_train_obs = torch.tensor(
        gaussian_filter1d(y_train_obs.cpu(), sigma=cfg.gaussian_kernel_sz//cfg.bin_sz_ms, axis=1)
    )#.to(cfg.data_device)
    y_valid_obs = torch.tensor(
        gaussian_filter1d(y_valid_obs.cpu(), sigma=cfg.gaussian_kernel_sz//cfg.bin_sz_ms, axis=1)
    )#.to(cfg.data_device)
    y_test_obs = torch.tensor(
        gaussian_filter1d(y_test_obs.cpu(), sigma=cfg.gaussian_kernel_sz//cfg.bin_sz_ms, axis=1)
    )#.to(cfg.data_device)

    vel_train = torch.stack((train_data['cursor_vel_x'], train_data['cursor_vel_y']), dim=-1)[:, :cfg.n_bins_enc, :]
    vel_valid = torch.stack((val_data['cursor_vel_x'], val_data['cursor_vel_y']), dim=-1)[:, :cfg.n_bins_enc, :]
    vel_test = torch.stack((test_data['cursor_vel_x'], test_data['cursor_vel_y']), dim=-1)[:, :cfg.n_bins_enc, :]

    y_train_dataset = torch.utils.data.TensorDataset(y_train_obs, vel_train)
    y_val_dataset = torch.utils.data.TensorDataset(y_valid_obs, vel_valid)
    y_test_dataset = torch.utils.data.TensorDataset(y_test_obs, vel_test)

    train_dataloader = torch.utils.data.DataLoader(y_train_dataset, batch_size=cfg.batch_sz, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(y_val_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(y_test_dataset, batch_size=y_valid_obs.shape[0], shuffle=False)

    n_train_trials, n_bins, n_neurons_obs = y_train_obs.shape
    n_valid_trials = y_valid_obs.shape[0]
    n_test_trials = y_test_obs.shape[0]

#####################################################################################################

    """likelihood pdf"""
    R_diag = torch.ones(n_neurons_obs, device=cfg.device)
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, nn.Linear(cfg.n_latents_read, n_neurons_obs))
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons_obs, R_diag, device=cfg.device)

    """dynamics module"""
    Q_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)

    """initial condition"""
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1. * torch.ones(cfg.n_latents, device=cfg.device)
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)

    """local/backward encoder"""
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons_obs, cfg.n_hidden_local, cfg.n_latents,
                                      rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    nl_filter = NonlinearFilter(dynamics_mod, initial_condition_pdf, device=cfg.device)

    """sequence vae"""
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, backward_encoder,
                                          local_encoder, nl_filter, device=cfg.device)
    
    """lightning"""
    model_ckpt_path = 'ckpts/smoother/acausal/last.ckpt'
    seq_vae = LightningNonlinearSSM.load_from_checkpoint(model_ckpt_path, ssm=ssm, cfg=cfg, n_time_bins_enc=cfg.n_bins_enc, n_time_bins_bhv=cfg.n_bins_bhv, strict=False)
    #seq_vae = LightningNonlinearSSM(ssm, cfg)

    csv_logger = CSVLogger('logs/smoother/acausal/',
                           name=f'sd_{cfg.seed}_r_y_{cfg.rank_local}_r_b_{cfg.rank_backward}',
                           version='v1')

    ckpt_callback = ModelCheckpoint(save_top_k=3, monitor='valid_loss', mode='min',
                                    dirpath='ckpts/smoother/acausal/', save_last=True,
                                    filename='{epoch:0}_{valid_loss:0.2f}')

    trainer = lightning.Trainer(accelerator="gpu",
                                strategy="ddp",
                                max_epochs=cfg.n_epochs,
                                gradient_clip_val=1.0,
                                default_root_dir='lightning/',
                                callbacks=[ckpt_callback],
                                logger=csv_logger,
                                )

    trainer.fit(model=seq_vae, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    torch.save(ckpt_callback.best_model_path, 'ckpts/smoother/acausal/best_model_path.pt')
    trainer.test(dataloaders=test_dataloader, ckpt_path='last')


if __name__ == '__main__':
    main()
