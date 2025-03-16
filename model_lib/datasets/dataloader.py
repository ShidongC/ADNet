

def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    pre_seq_length = kwargs.get('pre_seq_length', 12)
    aft_seq_length = kwargs.get('aft_seq_length', 12)

    if dataname == 'wind_10m_area1':  
        from .dataloader_wind_10m_area1 import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
