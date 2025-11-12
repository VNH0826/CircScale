# data/dataloader.py
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from config import Config


def create_dataloaders(dataset, batch_size, train_ratio=0.66, val_ratio=0.33):
    """
    Recipe-Inductive
    """
    all_recipes = sorted(set(dataset.df['Recipe'].tolist()))
    n_recipes = len(all_recipes)

    # 按recipe划分
    n_train_recipes = int(n_recipes * train_ratio)  # 66%的recipe用于训练
    train_recipes = set(all_recipes[:n_train_recipes])
    val_recipes = set(all_recipes[n_train_recipes:])  # 剩余34%用于验证

    print(f"Total recipes: {n_recipes}")
    print(f"Train recipes: {len(train_recipes)}, Val recipes: {len(val_recipes)}")

    # 根据recipe划分数据
    train_indices = []
    val_indices = []

    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        recipe = row['Recipe']

        if recipe in train_recipes:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # 创建子数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    kwargs = {
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY,
        'persistent_workers': True if Config.NUM_WORKERS > 0 else False
    }

    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader