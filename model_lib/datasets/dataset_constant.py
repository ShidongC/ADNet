dataset_parameters = {   
    'wind_10m_area1': {  # # u10+v10, component_of_wind, 
        'in_shape': [12, 2, 64, 64],
        'pre_seq_length': 12,
        'aft_seq_length': 12,
        'total_length': 24,
        'data_name': 'wind_10m_area1',
        'train_time': ['2019', '2020'], 'val_time': ['2021'], 'test_time': ['2022'],
    },
}