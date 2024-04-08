import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

def extract_hours(x : pd.Series):
    h = x.apply(lambda x : x.split(':')[0])
    return h

def tpv_preprocess(df : pd.DataFrame):
    df = df.loc[(df['Time'].shift() != df['Time']) | ~df['Time'].duplicated()]
    time_range = pd.DataFrame({'Time': range(48)})
    df = df.set_index('Time').join(time_range.set_index('Time'), how='outer').fillna(0)
    return df

def get_ground_truth(observed_masks : pd.Series, missing_ratio = 0.1):
    gt = observed_masks.copy()
    n_missing = int(missing_ratio * len(gt))
    missing_indices = np.random.choice(gt.index, n_missing, replace=False)
    gt[missing_indices] = 0
    return gt

def extract_file(attributes, root_dir, txt_id, missing_ratio = 0.1):
    df = pd.read_csv(os.path.join(root_dir, txt_id))
    df['Time'] = extract_hours(df['Time']).apply(lambda x : int(x))
    
    value_df, observed_masks_df, gt_masks_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for attr in attributes:
        attr_df = df.loc[lambda x : x['Parameter'] == attr, ['Time', 'Value']]
        attr_df = tpv_preprocess(attr_df)
        attr_df["observed_masks"] = (attr_df['Value'] != 0).astype('int')
        value = attr_df.loc[:, 'Value']
        observed_masks = attr_df.loc[:, 'observed_masks']
        value_df[attr], observed_masks_df[attr], gt_masks_df[attr] = value, observed_masks, get_ground_truth(observed_masks, missing_ratio = missing_ratio)
    
    return value_df.to_numpy(), observed_masks_df.to_numpy(), gt_masks_df.to_numpy()

def get_id_list(path):
    file_ids = []
    for filename in os.listdir(path):
        file_ids.append(filename)
    return file_ids[1:]

def normalized_value(observed_value:np.array, observed_masks:np.array, feature_num:int):
    tmp_value, tmp_masks = observed_value.reshape(-1, feature_num), observed_masks.reshape(-1, feature_num)
    mean, std = np.zeros(feature_num), np.zeros(feature_num)
    
    for i in range(feature_num):
        data = tmp_value[:, i][tmp_masks[:, i] == 1]
        mean[i], std[i] = data.mean(), data.std()

    observed_value = (
        (observed_value - mean) / std * observed_masks
    )

    return observed_value

class Physio_Dataset(Dataset):
    def __init__(self, root_dir, eval_length = 48, index_list = None, missing_ratio = 0.1, seed= 54):

        attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']
        
        np.random.seed(seed)

        path = ("C:/Users/DMQA/DMQA_documents/DMQA_documents/2024-1/Seminar/Mentoring/CSDI_practice/dataset/" + str(missing_ratio) + '_seed' + str(seed) + '.pk')

        self.eval_length = eval_length

        if os.path.isfile(path) == False:
            self.id_list = get_id_list(root_dir)
            self.obs_list, self.obs_mask_list, self.gt_mask_list, self.time_points_list = [], [], [], []
            
            for id in self.id_list:               
                v, o, g = extract_file(attributes, root_dir, id, missing_ratio = missing_ratio)
                print('extracting text file from id number {}...'.format(id))
                if index_list:
                    v, o, g = v[index_list, :], o[index_list, :], g[index_list, :]
                    self.time_points_list.append(len(index_list))
                else:
                    self.time_points_list.append(np.arange(self.eval_length))
                self.obs_list.append(v[:eval_length])
                self.obs_mask_list.append(o[:eval_length])
                self.gt_mask_list.append(g[:eval_length])

            self.observed_value, self.observed_masks, self.gt_masks = np.array(self.obs_list), np.array(self.obs_mask_list), np.array(self.gt_mask_list)
            self.observed_value = normalized_value(self.observed_value, self.observed_masks, 35)

            with open(path, "wb") as f:
                pickle.dump([self.observed_value, self.observed_masks, self.gt_masks, self.time_points_list] ,f)
                print("Successfully save the dataset")

        else:
            with open(path, "rb") as f:
                self.observed_value, self.observed_masks, self.gt_masks, self.time_points_list = pickle.load(
                    f
                )

        if index_list is None:
            self.index_list = np.arange(len(self.observed_value))
        else: 
            self.index_list = index_list

    def __getitem__(self, chosen_idx):
        item = {
            'observed_data': self.observed_value[chosen_idx],
            'observed_mask': self.observed_masks[chosen_idx],
            'gt_mask' : self.gt_masks[chosen_idx],
            # 'timepoints' : self.time_points_list[chosen_idx]
        }
        return item
    
    def __len__(self):
        return len(self.index_list)
    
def get_dataloader(root_dir, seed=54, nfold=None, batch_size=16, missing_ratio=0.1):
    """
    root_dir : Raw data 저장 위치 경로
    seed : 랜덤 시드
    nfold : 폴드 개수
    batch_size : 배치 개수
    missing_ratio : 마스크 missing 비율
    """

    dataset = Physio_Dataset(root_dir = root_dir,
                             eval_length = 48,
                             missing_ratio = missing_ratio,
                             seed = seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = Physio_Dataset(
        root_dir = root_dir, index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        root_dir = root_dir, index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Physio_Dataset(
        root_dir = root_dir, index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    train, valid, test = get_dataloader(root_dir = 'C:/Users/DMQA/DMQA_documents/DMQA_documents/2024-1/Seminar/Mentoring/CSDI_practice/dataset/set-a',
                                        seed = 54,
                                        nfold = 3,
                                        batch_size = 16,
                                        missing_ratio = 0.1)
    print(next(iter(train)))
