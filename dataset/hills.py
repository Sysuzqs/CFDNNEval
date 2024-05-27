import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class IRHillsDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'irRE',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 data_delta_time = 0.1,
                 delta_time: float = 0.1,   
                 num_samples_max = -1,
                 stable_state_diff = 0.0001,
                 norm_props = True,
                 norm_bc = True,
                 reshape_parameters = True,
                 multi_step_size = 1,
                 ):
        self.time_step_size = int(delta_time / data_delta_time)
        self.case_name = case_name
        self.multi_step_size = multi_step_size
        self.inputs = []
        self.labels = []
        self.masks = []
        self.case_params = []
        self.grids = []
        self.case_ids = []

        # dataset statistic
        self.statistics = {}
        self.statistics['vel_x_min'] = -28.0815334320
        self.statistics['vel_x_max'] =  66.4669570923
        self.statistics['vel_y_min'] = -32.2173538208
        self.statistics['vel_y_max'] =  31.7274246216
        self.statistics['vel_z_min'] = -12.8192892075
        self.statistics['vel_z_max'] =  26.2205390930
        self.statistics['prs_min']   = -313.4261779785
        self.statistics['prs_max']   =  862.0435180664

        self.statistics['pos_x_min'] = 0.0   # left bound
        self.statistics['pos_x_max'] = 9.0
        self.statistics['pos_y_min'] = 0.0   # lower bound
        self.statistics['pos_y_max'] = 4.5
        self.statistics['pos_z_min'] = 0.0   # lower bound
        self.statistics['pos_z_max'] = 3.035

        self.statistics['x_len'] = self.statistics['pos_x_max'] - self.statistics['pos_x_min']
        self.statistics['y_len'] = self.statistics['pos_y_max'] - self.statistics['pos_y_min']
        self.statistics['z_len'] = self.statistics['pos_z_max'] - self.statistics['pos_z_min']

        root_path = os.path.join(saved_folder, filename)
        with h5py.File(root_path, 'r') as f:
            case_id = 0
            for name in f.keys():
                if name not in case_name.split('_'):
                    continue
                
                case_dataset = f[name]
                data_keys = sorted(case_dataset.keys())[::reduced_batch]
                for case in data_keys:
                    # load case data
                    data = case_dataset[case]

                    # u, v, w, p
                    u = torch.from_numpy(data['Vx'][::reduced_resolution, ::self.time_step_size].transpose(1, 0)) # [T, nx]
                    v = torch.from_numpy(data['Vy'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))
                    w = torch.from_numpy(data['Vz'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))
                    p = torch.from_numpy(data['P'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))
                    
                    # normalize the input (comment on this part of code if normalized data explicitly in train loop)
                    u = (u - self.statistics['vel_x_min']) / (self.statistics['vel_x_max'] - self.statistics['vel_x_min'])
                    v = (v - self.statistics['vel_y_min']) / (self.statistics['vel_y_max'] - self.statistics['vel_y_min'])
                    w = (w - self.statistics['vel_z_min']) / (self.statistics['vel_z_max'] - self.statistics['vel_z_min'])
                    p = (p - self.statistics['prs_min']) / (self.statistics['prs_max'] - self.statistics['prs_min'])

                    # grid
                    grid = data['grid'][::reduced_resolution]

                    # normalize the grid
                    grid[:,0] = grid[:,0] / self.statistics['pos_x_max']
                    grid[:,1] = grid[:,1] / self.statistics['pos_y_max']
                    grid[:,2] = grid[:,2] / self.statistics['pos_z_max']


                    self.grids.append(torch.from_numpy(grid)) # [nx, 3]

                    # case param (RE here)
                    case_params = {"RE": data["RE"][0]}
                    
                    # mask
                    masks = torch.ones_like(u).unsqueeze(-1) # [T, nx, 1]
                
                    case_features = torch.stack((u, v, w, p), dim=-1)
                    inputs = case_features[:-self.multi_step_size] # [T-self.time_step_size, nx, 1]

                    for i in range(inputs.shape[0]):
                        self.inputs.append(inputs[i])
                        self.labels.append(case_features[i+1: i+1+self.multi_step_size])
                        self.case_ids.append(case_id)
                        self.masks.append(masks[i+1: i+1+self.multi_step_size])

                    # normalize case parameters
                    if norm_props:
                        self.normalize_physics_props(case_params)

                    params_keys = ['RE']
                    case_params_vec = []
                    for k in params_keys:
                        case_params_vec.append(case_params[k])
                    self.case_params.append(torch.tensor(case_params_vec, dtype=torch.float32))

                    case_id += 1

        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)
        self.masks = torch.stack(self.masks)
        self.case_ids = torch.tensor(self.case_ids)

        if multi_step_size == 1:
            self.labels.squeeze_(1)
            self.masks.squeeze_(1)

        if num_samples_max > 0:
            assert num_samples_max < self.inputs.shape[0]
            self.inputs = self.inputs[:num_samples_max]
            self.labels = self.labels[:num_samples_max]
            self.case_ids = self.case_ids[:num_samples_max]
            self.masks = self.masks[:num_samples_max]
        
    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        case_params["RE"] = (
            case_params["RE"] - 505.6250000000 
        ) / 299.4196166992

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        grid = self.grids[case_id]
        nx = grid.shape[0]
        case_params = self.case_params[case_id].unsqueeze(0).repeat([nx, 1])
        return self.inputs[idx], self.labels[idx], self.masks[idx], case_params, grid, case_id