import torch
import torch.nn as nn
import os.path as osp
import copy
from model_lib.modules.adnet_modules import ADNetBlock
from model_lib.modules.constraint_modules import K2M


class ADNet_Model(nn.Module):

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ADNet_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        cell_list.append(
            ADNetBlock(self.frame_channel, num_hidden[0], height, width, 5, 1)
        )
        cell_list.append(
            ADNetBlock(num_hidden[1], num_hidden[1], height, width, 5, 1)
        )

        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(num_hidden[0], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        self.k2m = K2M([configs.filter_size,configs.filter_size])
        self.constraints_dx, self.constraints_dy, self.constraints_dxdx, self.constraints_dydy = self._get_constraints()

    def _get_constraints(self):
        constraints_dx = torch.zeros((1, self.configs.filter_size, self.configs.filter_size)).to(self.configs.device)
        constraints_dy = torch.zeros((1, self.configs.filter_size, self.configs.filter_size)).to(self.configs.device)
        constraints_dx[0,0,1] = 1
        constraints_dx = constraints_dx.repeat(self.num_hidden[0], 1, 1)
        constraints_dy[0,1,0] = 1
        constraints_dy = constraints_dy.repeat(self.num_hidden[0], 1, 1)

        constraints_dxdx = torch.zeros((1, self.configs.filter_size, self.configs.filter_size)).to(self.configs.device)
        constraints_dydy = torch.zeros((1, self.configs.filter_size, self.configs.filter_size)).to(self.configs.device)
        constraints_dxdx[0,0,2] = 1
        constraints_dxdx = constraints_dxdx.repeat(self.num_hidden[0], 1, 1)
        constraints_dydy[0,2,0] = 1
        constraints_dydy = constraints_dydy.repeat(self.num_hidden[0], 1, 1)

        return constraints_dx, constraints_dy, constraints_dxdx, constraints_dydy

    def forward(self, frames_tensor, mask_true, is_train=False):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height//self.configs.stride, width//self.configs.stride]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling

            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                        (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t_0 = h_t[0]
            h_t_1 = h_t[1]
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0], h_t_1)
            h_t[1], c_t[1] = self.cell_list[1](h_t[0], h_t[1], c_t[1], h_t_0)

            x_gen = self.conv_last(h_t[0] + h_t[1])
            next_frames.append(x_gen)

        loss_PDE = 0
        for a in range(self.num_layers):
            filters_dx = self.cell_list[a].dx.weight[:,0,:,:]
            m_dx = self.k2m(filters_dx.double()).float()
            filters_dy = self.cell_list[a].dy.weight[:,0,:,:]
            m_dy = self.k2m(filters_dy.double()).float()

            filters_dxdx = self.cell_list[a].dxdx.weight[:,0,:,:]
            m_dxdx = self.k2m(filters_dxdx.double()).float()
            filters_dydy = self.cell_list[a].dydy.weight[:,0,:,:]
            m_dydy = self.k2m(filters_dydy.double()).float()

            loss_PDE += self.MSE_criterion(m_dx, self.constraints_dx)
            loss_PDE += self.MSE_criterion(m_dy, self.constraints_dy)
            loss_PDE += self.MSE_criterion(m_dxdx, self.constraints_dxdx)
            loss_PDE += self.MSE_criterion(m_dydy, self.constraints_dydy)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        if is_train:
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + loss_PDE
        else:   
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])

        return next_frames, loss
