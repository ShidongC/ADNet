import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def metric(pred, true, mean, std, metrics=['mae', 'mse'],
           clip_range=[0, 1], spatial_norm=False):
  
    pred = pred * std + mean
    true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim',]
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, true, spatial_norm)

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, true, spatial_norm)

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, true, spatial_norm)

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    for k, v in eval_res.items():
        eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
        eval_log += eval_str

    return eval_res, eval_log

def metric_2(pred, true, mean, std, metrics=['mae', 'mse'],
           clip_range=[0, 1], spatial_norm=False):

    pred = pred * std + mean
    true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'mae_frame', 'mse_frame', 'rmse_frame', 'ssim_frame', 'R2']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    if 'mae' in metrics:
        norm = pred.shape[-1] * pred.shape[-2]
        eval_res['mae_frame'] = np.mean(np.abs(pred-true) / norm, axis=(0)).sum(axis=(-1,-2))
        eval_res['mae'] = np.mean(np.abs(pred-true) / norm, axis=(0,1)).sum(axis=(-1,-2))

    if 'mse' in metrics:
        norm = pred.shape[-1] * pred.shape[-2]
        eval_res['mse_frame'] = np.mean((pred-true)**2 / norm, axis=(0)).sum(axis=(-1,-2))
        eval_res['mse'] = np.mean((pred-true)**2 / norm, axis=(0,1)).sum(axis=(-1,-2))

    if 'rmse' in metrics:
        norm = pred.shape[-1] * pred.shape[-2]
        eval_res['rmse_frame'] = np.sqrt(np.mean((pred-true)**2 / norm, axis=(0)).sum(axis=(-1,-2)))
        eval_res['rmse'] = np.sqrt(np.mean((pred-true)**2 / norm, axis=(0,1)).sum(axis=(-1,-2)))
    
    if 'R2' in metrics:
        true_mean = np.mean(true, axis=(0,1,3,4), keepdims=True)
        up = np.sum((pred-true)**2, axis=(0,1,3,4))
        down = np.sum((true_mean-true)**2, axis=(0,1,3,4))
        eval_res['R2'] =  1 - up / down

    max = np.maximum(np.max(pred), np.max(true))
    min = np.minimum(np.min(pred), np.min(true))
    pred = (pred - min)/(max - min)
    true = (true - min)/(max - min)

    if 'ssim' in metrics:
        ssim = np.zeros(2)
        ssim_frame = np.zeros((pred.shape[1],2))
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                for c in range(pred.shape[2]):
                    ssim[c] += cal_ssim(pred[b, f, c],true[b, f, c])
                    ssim_frame[f, c] += cal_ssim(pred[b, f, c],true[b, f, c])

        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])
        eval_res['ssim_frame'] = ssim_frame/pred.shape[0]

    for k, v in eval_res.items():
        eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
        eval_log += eval_str

    return eval_res, eval_log