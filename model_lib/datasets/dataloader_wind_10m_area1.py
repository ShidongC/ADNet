##areas1 is 2N_17.75N_105E_120.75E

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset

def get_data(data_root, variable, years):
    for year in years:
        frames_year = np.load(data_root + variable + '/' + year + '_' + variable + '.npy')
        if year == years[0]:
            frames = frames_year
        else:
            frames = np.concatenate((frames, frames_year), axis=0)   
    return frames

class WindDataset1(Dataset):

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step,
                 mean=None, std=None,
                 transform_data=None, transform_labels=None):
        super().__init__()
        self.dataname = data_name
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels

        self.time = None

        u_frames = get_data(data_root, '10m_u_component_of_wind', self.training_time)
        v_frames = get_data(data_root, '10m_v_component_of_wind', self.training_time)
        self.data = np.stack((u_frames, v_frames), axis=1)

        if self.mean is None:
            self.mean = self.data.mean(axis=(0, 2, 3)).reshape(
                1, self.data.shape[1], 1, 1)
            self.std = self.data.std(axis=(0, 2, 3)).reshape(
                1, self.data.shape[1], 1, 1)

        self.data = (self.data-self.mean)/self.std

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1, self.step))

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in], dtype=torch.float32)
        labels = torch.tensor(self.data[index+self.idx_out], dtype=torch.float32)
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_name='wind_10m_area1',
              train_time=['2019', '2020'],
              val_time=['2021'],
              test_time=['2022'],
              idx_in=[-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0],
              idx_out=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              step=6,
              **kwargs):

    wind_dataroot = data_root + '2N_17.75N_105E_120.75E/'

    train_set = WindDataset1(data_root=wind_dataroot,
                               data_name=data_name,
                               training_time=train_time,
                               idx_in=idx_in,
                               idx_out=idx_out,
                               step=step)
    validation_set = WindDataset1(wind_dataroot,
                                    data_name,
                                    val_time,
                                    idx_in,
                                    idx_out,
                                    step,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WindDataset1(wind_dataroot,
                              data_name,
                              test_time,
                              idx_in,
                              idx_out,
                              step,
                              mean=train_set.mean,
                              std=train_set.std)

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_vali = torch.utils.data.DataLoader( validation_set, # validation_set,
                                                   batch_size=val_batch_size, shuffle=False,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                   batch_size=val_batch_size, shuffle=False,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers)

    return dataloader_train, dataloader_vali, dataloader_test