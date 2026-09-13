import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Dict

class TimeSeriesWindowDataset(Dataset):
    """
    Custom PyTorch Dataset for 3D Sliding Window representations.
    Input Tensor: (Samples, Timesteps, Features)
    Target: Next-day return r_{t+1} (Scalar)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_windows(
    data: np.ndarray, 
    target_idx: int, 
    window_size: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3D tensors without data leakage.
    Sample i: features from [i : i + window_size]
    Label i: target at [i + window_size] (corresponding to t+1)
    """
    X, y = [], []
    num_records = len(data)
    for i in range(num_records - window_size):
        X.append(data[i : i + window_size, :])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def prepare_dataloaders(
    csv_path: str,
    target_col: str = 'Gold_Price_Return',
    window_size: int = 21,
    batch_size: int = 32,
    train_end_date: str = '2021-12-31',
    val_end_date: str = '2023-12-31'
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray], StandardScaler]:
    """
    Complete Pipeline:
    1. Load daily_processed_features.csv
    2. Chronological Split (Train / Val / Test)
    3. Fit Scaler ONLY on the Train set
    4. Extract 3D sliding window arrays
    5. Package into DataLoaders
    """
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    target_idx = df.columns.get_loc(target_col)

    # 1. Chronological splitting by date
    df_train = df.loc[df.index <= train_end_date].copy()
    df_val = df.loc[(df.index > train_end_date) & (df.index <= val_end_date)].copy()
    df_test = df.loc[df.index > val_end_date].copy()

    print(f"Data Split Summary:")
    print(f"- Train : {df_train.index.min().date()} to {df_train.index.max().date()} | Rows: {len(df_train)}")
    print(f"- Val   : {df_val.index.min().date()} to {df_val.index.max().date()} | Rows: {len(df_val)}")
    print(f"- Test  : {df_test.index.min().date()} to {df_test.index.max().date()} | Rows: {len(df_test)}")

    # 2. Fit Scaler ONLY on the Train set to prevent Look-Ahead Bias
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)

    # To ensure window continuity at the boundary between splits,
    # prepend the last window_size days of the previous set to the beginning of the next set
    val_padded = np.vstack([train_scaled[-window_size:], val_scaled])
    test_padded = np.vstack([val_scaled[-window_size:], test_scaled])

    # 3. Create Sliding Windows (Samples, Window_Size, Features)
    X_train, y_train = create_sliding_windows(train_scaled, target_idx, window_size)
    X_val, y_val = create_sliding_windows(val_padded, target_idx, window_size)
    X_test, y_test = create_sliding_windows(test_padded, target_idx, window_size)

    print(f"\nTensor Shapes:")
    print(f"- X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"- X_val  : {X_val.shape}, y_val: {y_val.shape}")
    print(f"- X_test : {X_test.shape}, y_test: {y_test.shape}")

    # 4. Package PyTorch Datasets & DataLoaders
    train_dataset = TimeSeriesWindowDataset(X_train, y_train)
    val_dataset = TimeSeriesWindowDataset(X_val, y_val)
    test_dataset = TimeSeriesWindowDataset(X_test, y_test)

    # Train can shuffle windows to mitigate local overfitting; Val/Test retain temporal order
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    raw_arrays = {
        'X_test': X_test,
        'y_test': y_test,
        'test_dates': df_test.index.values
    }

    return dataloaders, raw_arrays, scaler


if __name__ == "__main__":
    # Test the data loading pipeline
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_file = base_dir / "data" / "processed" / "daily_processed_features.csv"
    
    if data_file.exists():
        loaders, raw, sc = prepare_dataloaders(str(data_file))
        sample_x, sample_y = next(iter(loaders['train']))
        print(f"\nBatch inspection:")
        print(f"Batch X shape: {sample_x.shape} (Batch_size, Window, Features)")
        print(f"Batch y shape: {sample_y.shape} (Batch_size, 1)")
    else:
        print(f"File not found: {data_file}")