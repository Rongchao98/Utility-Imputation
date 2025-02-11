import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os


def get_dataloader_s1(config, subsequence=False, nrows=None):
    df, mean, std, users = process_raw_data(config['raw_df_path'], nrows=nrows)
    observed_data, observed_mask, gt_mask, index_month, position_in_month, position_in_week, _, _, split = preprocess_data(df, users, config['gt_path'], config['split_path'], nrows=nrows)
    dataset = HistoryTsDataset(config, observed_data, split, subsequence, 0)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return data_loader


def get_dataloader_s2(config, i_vae, l_vae, nrows=None):
    df, mean, std, users = process_raw_data(config['raw_df_path'], nrows=nrows)
    observed_data, observed_mask, gt_mask, index_month, position_in_month, position_in_week, _, _, split = preprocess_data(
        df, users, config['gt_path'], config['split_path'], nrows=nrows)
    dataset = UtilityDataset(config, observed_data, observed_mask, gt_mask, index_month, position_in_month, position_in_week,
                             users, mean, std, i_vae, l_vae)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    return data_loader


def process_raw_data(path_raw, nrows=None):
    df = pd.read_csv(
        path_raw,
        parse_dates=True,
        nrows=nrows,
    )
    df['DATE_RD'] = pd.to_datetime(df['DATE_RD'])
    mean = df.fillna(0).iloc[:, 2:50].mean().values
    std = df.fillna(0).iloc[:, 2:50].std().values
    users = df['BADGE'].unique()
    return df, mean, std, users


def split_data(df, path_split, split_rate=(0.7, 0.1, 0.2)):
    split = []  # split code (separated into each user)
    users = df['BADGE'].unique()

    for user in users:
        num = df[df['BADGE'] == user].shape[0]
        # the first one belongs to train, the rest split by split_rate;
        u_split_df = np.zeros(num)
        u_split_df[0] = 0
        u_split_df[1:] = np.random.choice([0, 1, 2], num - 1, p=split_rate)
        # transform user_split into a df: BADGE, split
        u_split_df = pd.DataFrame({
            'BADGE': [user] * num,
            'split': u_split_df,
        })
        # assign the type of split to int
        u_split_df['split'] = u_split_df['split'].astype(int)
        split.append(u_split_df)

    split_df = pd.concat(split)
    split_df.to_csv(path_split, index=False)

def distance(v, k):
    return np.linalg.norm(v - k, axis=1)

def preprocess_data(
        df, # raw df
        users, # list of users
        path_gt="./data/utility/utility_missing.txt",
        path_split = "./data/utility/split.csv",
        nrows=None,
    ):
    # create data for batch
    observed_data = []  # values (separated into each user)
    observed_mask = []  # masks (separated into each user)
    gt_mask = []  # ground-truth masks (separated into each user)
    index_month = []  #  month (separated into each user)
    position_in_month = []  # position in month (separated into each user)
    position_in_week = []  # position in week (separated into each user)
    district = []  # district (separated into each user)
    location = []  # location (GPS) (separated into each user)
    split = []  # split code (separated into each user)

    df_gt = pd.read_csv(
        path_gt,
        parse_dates=True,
        nrows=nrows,
    )
    df_gt['DATE_RD'] = pd.to_datetime(df_gt['DATE_RD'])

    for user in users:
        u_df = df[df['BADGE'] == user]
        u_df_gt = df_gt[df_gt['BADGE'] == user]

        u_ts_df = u_df.iloc[:, 2:50]
        u_ts_df_gt = u_df_gt.iloc[:, 2:50]
        u_district_df = u_df['address'].values
        u_location_df = u_df['district'].values

        c_mask = 1 - u_ts_df.isnull().values
        c_gt_mask = 1 - u_ts_df_gt.isnull().values
        c_data = (
            (u_ts_df.fillna(0).values - mean) / std
        ) * c_mask
        c_month = u_df['DATE_RD'].dt.month.to_numpy()
        c_ps_in_month = u_df['DATE_RD'].dt.day.to_numpy()
        c_ps_in_week = u_df['DATE_RD'].dt.weekday.to_numpy()

        observed_mask.append(c_mask)
        gt_mask.append(c_gt_mask)
        observed_data.append(c_data)
        index_month.append(c_month)
        position_in_month.append(c_ps_in_month)
        position_in_week.append(c_ps_in_week)
        district.append(u_district_df)
        location.append(u_location_df)

    if os.path.isfile(path_split) and path_split.endswith('.csv'):
        df_split = pd.read_csv(
            path_split,
            index_col=['BADGE'],
        )

        df_split['split'] = df_split['split'].astype(int)
        for user in users:
            split.append(df_split.loc[user, 'split'].values)
    else:
        # print error
        print("Error: split file not found")
        return

    return (observed_data,
            observed_mask,
            gt_mask,
            index_month,
            position_in_month,
            position_in_week,
            district,
            location,
            split,)


class HistoryTsDataset(Dataset):
    def __init__(self,
            config,
            observed_data, # list of ndarray
            flag,       # list of ndarray
            subsequence=False, # if True, return subsequence
            train_code=0,
            ):
        self.his_data = observed_data
        self.flag = flag
        self.train_code = train_code

        data = []
        for i, u_code in enumerate(self.flag):
            indices = np.where(u_code == self.train_code)[0]
            t_data = self.his_data[i][indices]

            if subsequence:
                # split each row in data into subsequence
                # with size config['subsequence_length'] and stride config['subsequence_stride']
                # and append to data
                for j in range(0, t_data.shape[0] - config['subsequence_length'] + 1, config['subsequence_stride']):
                    data.append(t_data[j:j+config['subsequence_length']])
            else:
                data.append(t_data)

        self.data = np.concatenate(data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class UtilityDataset(Dataset):
    def __init__(self,
            config,
            observed_data,
            observed_mask,
            gt_mask,
            index_month,
            position_in_month,
            position_in_week,
            users,
            train_mean,
            train_std,
            i_vae,
            l_vae,):

        self.observed_data = observed_data
        self.observed_mask = observed_mask
        self.gt_mask = gt_mask

        self.train_mean = train_mean
        self.train_std = train_std

        self.month = index_month
        self.ps_month = position_in_month
        self.ps_week = position_in_week
        self.i_vae = i_vae
        self.l_vae = l_vae

        self.users = users

        data = []
        for i, u_code in enumerate(self.flag):
            indices = np.where(u_code == self.train_code)[0]
            data.append(self.observed_data[i][indices])

        self.data = np.concatenate(data, axis=0)

        # 1. Global pattern
        gp_d, gp_m, gp_y = [], [], []
        for i in range(7):
            gp_d.append(self.data[self.ps_week == i])

        for i in range(1, 13):
            gp_m.append(self.data[self.month == i])

        for i in range(2015, 2021):
            gp_y.append(self.data[self.year == i])

        gp_d_mean = [np.mean(d, axis=0) for d in gp_d]
        gp_d_std = [np.std(d, axis=0) for d in gp_d]
        gp_m_mean = [np.mean(d, axis=0) for d in gp_m]
        gp_m_std = [np.std(d, axis=0) for d in gp_m]
        gp_y_mean = [np.mean(d, axis=0) for d in gp_y]
        gp_y_std = [np.std(d, axis=0) for d in gp_y]
        self.gp = [gp_d_mean, gp_d_std, gp_m_mean, gp_m_std, gp_y_mean, gp_y_std]

        # 2. Local pattern
        # split data into subsequences according to the length of local pattern
        data = []
        for i, u_data in enumerate(self.data):
            for j in range(0, u_data.shape[0] - config['subsequence_length'] + 1, config['subsequence_stride']):
                data.append(u_data[j:j+config['subsequence_length']])

        # get latent representation of each subsequence
        l_latents = []
        for i, u_data in enumerate(self.data):
            l_latent = self.l_vae.encode(torch.tensor(u_data).unsqueeze(0))
            l_latents.append(l_latent)

        # form the pair for each subsequence and its latent representation groupby position of subsequence
        positions = {}
        for i, u_data in enumerate(self.data):
            for j in range(0, u_data.shape[0] - config['subsequence_length'] + 1, config['subsequence_stride']):
                positions[(i, j)] = l_latents[i]

        self.lp = positions

        # 3. Instance pattern
        i_latents = []
        for i, u_data in enumerate(self.data):
            i_latent = self.i_vae.encode(torch.tensor(u_data).unsqueeze(0))
            i_latents.append(i_latent)

        i_d, i_m, i_y = [], [], []
        for i in range(7):
            i_d.append(i_latents[self.ps_week == i])

        for i in range(1, 13):
            i_m.append(i_latents[self.month == i])

        for i in range(2015, 2021):
            i_y.append(i_latents[self.year == i])

        i_d_matrix = np.array([
            [np.mean([distance(v, z) for v in i_latents[j] for z in i_latents[i]])]
            for j in range(7) for i in range(7)
        ]).reshape(7, 7)

        i_d_matrix = np.array([
            [(x - np.min(row)) / (np.max(row) - np.min(row)) for x in row]
            for row in i_d_matrix
        ])

        i_m_matrix = np.array([
            [np.mean([distance(v, z) for v in i_latents[j] for z in i_latents[i]])]
            for j in range(1, 13) for i in range(1, 13)
        ]).reshape(12, 12)

        i_m_matrix = np.array([
            [(x - np.min(row)) / (np.max(row) - np.min(row)) for x in row]
            for row in i_m_matrix
        ])

        i_y_matrix = np.array([
            [np.mean([distance(v, z) for v in i_latents[j] for z in i_latents[i]])]
            for j in range(2015, 2021) for i in range(2015, 2021)
        ]).reshape(6, 6)

        i_y_matrix = np.array([
            [(x - np.min(row)) / (np.max(row) - np.min(row)) for x in row]
            for row in i_y_matrix
        ])

        self.ip = [i_d_matrix, i_m_matrix, i_y_matrix]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get the data, mask, gt_mask, month, ps_month, ps_week, gp, lp, ip
        data = self.data[idx]
        mask = self.observed_mask[idx]
        gt_mask = self.gt_mask[idx]
        month = self.month[idx]
        ps_month = self.ps_month[idx]
        ps_week = self.ps_week[idx]
        gp = self.gp
        lp = self.lp[idx]
        ip = self.ip
        return data, mask, gt_mask, month, ps_month, ps_week, gp, lp, ip


if __name__ == "__main__":
    dataset_name = 'water'
    orginal_df_path = 'data/%s.csv' % dataset_name
    missing_df_dir = 'data/%s/missing_data' % dataset_name
    split_dir = 'data/%s/split' % dataset_name
    nrows = 10000
    limits = 50
    split_rate = (0.7, 0.1, 0.2)

    missing_df_prefix = dataset_name + '_missing'
    split_prefix = dataset_name + '_split'

    ratio_ls = [0.2, 0.35, 0.5]
    type_ls = ['Point', 'Block']

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    df, mean, std, users = process_raw_data(orginal_df_path, nrows=nrows)
    df = df.groupby('BADGE').filter(lambda x: len(x) >= limits)

    for ratio in ratio_ls:
        for type in type_ls:
            print("Processing %s with ratio %f" % (type, ratio))
            m_data_path_name = os.path.join(missing_df_dir, missing_df_prefix)
            path_gt = m_data_path_name + '_' + type + '_' + str(ratio) + '.csv'
            path_split = os.path.join(split_dir, split_prefix + '_' + type + '_' + str(ratio) + '.csv')
            split_data(df, path_split, split_rate)

    # dataset = HistoryTsDataset(observed_data, split, 0)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch in data_loader:
    #     print(batch.shape)
    #     break



