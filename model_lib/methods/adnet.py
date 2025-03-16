import torch
import torch.nn as nn
import numpy as np
from timm.utils import AverageMeter
from tqdm import tqdm

from model_lib.models.adnet_model import ADNet_Model
from model_lib.utils import (reshape_patch, reshape_patch_back,
                         reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method

class ADNet(Base_method):

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return ADNet_Model(num_layers, num_hidden, args).to(self.device)

    def test_one_epoch(self, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(test_loader)

        mask_input = self.args.pre_seq_length

        _, img_channel, img_height, img_width = self.args.in_shape

        for batch_x, batch_y in test_pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # preprocess
            test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            test_dat = reshape_patch(test_ims, self.args.patch_size)
            test_ims = test_ims[:, :, :, :, :img_channel]

            real_input_flag = torch.zeros(
                (batch_x.shape[0],
                self.args.total_length - mask_input - 1,
                img_height // self.args.patch_size,
                img_width // self.args.patch_size,
                self.args.patch_size ** 2 * img_channel)).to(self.device)

            img_gen, _ = self.model(test_dat, real_input_flag, is_train=False)
            img_gen = reshape_patch_back(img_gen, self.args.patch_size)
            pred_y = img_gen[:, -self.args.aft_seq_length:].permute(0, 1, 4, 2, 3).contiguous()

            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(
            lambda data: np.concatenate(data, axis=0), [inputs_lst, trues_lst, preds_lst])
        return inputs, trues, preds
