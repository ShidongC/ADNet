# Copyright (c) CAIRI AI Lab. All rights reserved

import time
import logging
import pickle
import json
import torch
import numpy as np
import pandas as pd
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

from model_lib.core import metric, metric_2, Recorder
from model_lib.methods import method_maps
from model_lib.utils import (set_seed, print_log, output_namespace, check_dir,
                         get_dataset, measure_throughput, weights_to_cpu)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


class NonDistExperiment(object):
    """ Experiment with non-dist PyTorch training and evaluation """

    def __init__(self, args):
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()
        self.args.method = self.args.method.lower()
        self._epoch = 0

        self._preparation()
        print_log(output_namespace(self.args))

        T, C, H, W = self.args.in_shape

        if self.args.method in ['adnet']:
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(2, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(2, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)

        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        print_log(self.method.model)
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        print_log(flop_count_table(flops))
        if args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            print_log('Throughputs of {}: {:.3f}'.format(self.args.method, fps))

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:', device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.dataname, self.args.method, self.args.round)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        prefix = 'train' if not self.args.test else 'test'
        logging.basicConfig(level=logging.INFO,
                            filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the method
        self._build_method()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self):
        self.train_loader, self.vali_loader, self.test_loader = get_dataset(self.args.dataname, self.config)
        if self.vali_loader is None:
            self.vali_loader = self.test_loader

    def _save(self, name=''):
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self.method.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def test(self):
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            self.method.model.load_state_dict(torch.load(best_model_path))

        inputs, trues, preds = self.method.test_one_epoch(self.test_loader)

        if self.args.dataname == 'wind_10m_area1':
            metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            metric_list, spatial_norm = ['mse', 'mae', 'ssim'], False
        eval_res, eval_log = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        metrics = np.array([eval_res['mae'], eval_res['mse']])
        print_log(eval_log)

        folder_path = osp.join(self.path, 'saved')
        check_dir(folder_path)

        # for np_data in ['metrics', 'inputs', 'trues', 'preds']:
        #     np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return eval_res['mse']


    def test_2(self):
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            self.method.model.load_state_dict(torch.load(best_model_path))

        inputs, trues, preds = self.method.test_one_epoch(self.test_loader)

        if self.args.dataname == 'wind_10m_area1':
            metric_list, spatial_norm = ['mse', 'rmse', 'mae', 'R2'], True
        else:
            metric_list, spatial_norm = ['mse', 'mae', 'ssim', 'psnr'], False
        eval_res, eval_log = metric_2(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        metrics = np.array([eval_res['mae'], eval_res['mse']])
        print_log(eval_log)

        folder_path = osp.join(self.path, 'saved')
        check_dir(folder_path)

        # for np_data in ['metrics', 'inputs', 'trues', 'preds']:
        #     np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return eval_res['mse']