import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Dict, List, Optional

class TimeSeriesWindowDataset(Dataset):
    """
    Standard PyTorch Dataset that accepts pre-sliced (X, y) arrays or tensors.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.clone().detach().to(dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y.clone().detach().to(dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_windows(
    data: np.ndarray, 
    target_idx: int, 
    dates: Optional[pd.DatetimeIndex] = None,
    window_size: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates 3D tensors strictly without data leakage.
    Sample i: features from [i : i + window_size] (t-20 to t)
    Label i: target at [i + window_size] (t+1)
    """
    X, y = [], []
    num_records = len(data)
    
    for i in range(num_records - window_size):
        # Target alignment verification if dates are provided
        if dates is not None:
            max_window_date = dates[i + window_size - 1]
            target_date = dates[i + window_size]
            assert max_window_date < target_date, (
                f"FATAL LEAKAGE: Window ends on {max_window_date}, but Target is {target_date}"
            )
            
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, target_idx])
        
    return np.array(X), np.array(y)


def prepare_dataloaders(
    csv_path: str,
    target_col: str = 'Gold_Price_Return',
    window_size: int = 21,
    batch_size: int = 32,
    train_end_date: str = '2021-12-31',
    val_end_date: str = '2023-12-31',
    keep_cols: Optional[List[str]] = None
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray], StandardScaler]:
    """
    Complete Data Pipeline:
    1. Chronological splitting (Train / Val / Test)
    2. Optional In-Memory Feature Filtering (for RQ3 Ablation)
    3. Fit Scaler ONLY on the Train set
    4. Slices sliding windows with target assertion
    5. Packages into PyTorch DataLoaders
    """
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # In-memory feature filtering (supporting RQ3 Nested Ablation)
    if keep_cols is not None:
        # Ensure target_col remains in the dataframe
        cols_to_use = list(dict.fromkeys(keep_cols + [target_col]))
        df = df[cols_to_use]

    target_idx = df.columns.get_loc(target_col)

    # 1. Chronological splitting by date
    df_train = df.loc[df.index <= train_end_date].copy()
    df_val = df.loc[(df.index > train_end_date) & (df.index <= val_end_date)].copy()
    df_test = df.loc[df.index > val_end_date].copy()

    print(f"Data Split Summary:")
    print(f"- Train : {df_train.index.min().date()} to {df_train.index.max().date()} | Rows: {len(df_train)}")
    print(f"- Val   : {df_val.index.min().date()} to {df_val.index.max().date()} | Rows: {len(df_val)}")
    print(f"- Test  : {df_test.index.min().date()} to {df_test.index.max().date()} | Rows: {len(df_test)}")

    # 2. Fit Scaler ONLY on the Train set
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)

    # Padded arrays to guarantee continuous window slicing across split boundaries
    val_padded = np.vstack([train_scaled[-window_size:], val_scaled])
    test_padded = np.vstack([val_scaled[-window_size:], test_scaled])

    # Date alignment tracking for padded datasets
    val_padded_dates = df_train.index[-window_size:].append(df_val.index)
    test_padded_dates = df_val.index[-window_size:].append(df_test.index)

    # 3. Create Sliding Windows with strict temporal assertions
    X_train, y_train = create_sliding_windows(train_scaled, target_idx, df_train.index, window_size)
    X_val, y_val = create_sliding_windows(val_padded, target_idx, val_padded_dates, window_size)
    X_test, y_test = create_sliding_windows(test_padded, target_idx, test_padded_dates, window_size)

    print(f"\nTensor Shapes:")
    print(f"- X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"- X_val  : {X_val.shape}, y_val: {y_val.shape}")
    print(f"- X_test : {X_test.shape}, y_test: {y_test.shape}")

    # 4. Package PyTorch Datasets & DataLoaders
    train_dataset = TimeSeriesWindowDataset(X_train, y_train)
    val_dataset = TimeSeriesWindowDataset(X_val, y_val)
    test_dataset = TimeSeriesWindowDataset(X_test, y_test)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    raw_arrays = {
        'X_train': X_train,
        'y_train': y_train,
        'train_dates': df_train.index.values,
        'X_val': X_val,
        'y_val': y_val,
        'val_dates': df_val.index.values,
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': df_test.index.values
    }

    return dataloaders, raw_arrays, scaler