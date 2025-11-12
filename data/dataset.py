# data/dataset.py
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


class CircScaleDataset(Dataset):
    def __init__(self, csv_path, graph_dir, heuristic_to_idx=None, train=True, metric_type="delay"):
        self.df = pd.read_csv(csv_path)
        self.graph_dir = graph_dir
        self.train = train
        self.metric_type = metric_type.lower()

        # 构建或使用启发式算法词汇表
        if heuristic_to_idx is None and train:
            self.build_heuristic_vocab()
        else:
            self.heuristic_to_idx = heuristic_to_idx

        self.normalize_qor()  # 归一化QoR值
        self.graph_cache = {}  

    def build_heuristic_vocab(self):
        """从所有Recipe中构建启发式算法词汇表"""
        all_heuristics = set()
        for recipe in self.df['Recipe']:
            heuristics = [h.strip() for h in recipe.split(';') if h.strip()]
            all_heuristics.update(heuristics)

        self.heuristic_to_idx = {h: i for i, h in enumerate(sorted(all_heuristics))}
        return self.heuristic_to_idx

    def normalize_qor(self):
        """标准化QoR值 - 根据指标类型选择不同的列名"""
        if self.metric_type == "area":
            # Area数据：Area_1到Area_20 (20步)
            qor_columns = [f'Area_{i + 1}' for i in range(20)]
            self.max_steps = 20
        else:
            # Delay数据：Level_1到Level_20 (20步)
            qor_columns = [f'Level_{i + 1}' for i in range(20)]
            self.max_steps = 20

        qor_values = self.df[qor_columns].values.astype(np.float32)

        self.qor_mean = np.mean(qor_values)
        self.qor_std = np.std(qor_values)

        # Store normalized values
        for col in qor_columns:
            self.df[f'{col}_norm'] = (self.df[col].astype(np.float32) - self.qor_mean) / self.qor_std

    def __len__(self):
        """返回数据集大小"""
        return len(self.df)

    def __getitem__(self, idx):
        """获取单个数据样本 - 使用原始QoR值"""
        row = self.df.iloc[idx]

        # 获取设计名称并加载对应的AIG图
        design_name = row['Design']
        graph = self.load_graph(design_name)

        # 处理Recipe
        recipe = row['Recipe'].split(';')
        recipe_indices = torch.tensor([self.heuristic_to_idx[h.strip()] for h in recipe if h.strip()], dtype=torch.long)

        # 根据指标类型获取QoR轨迹
        if self.metric_type == "area":
            qor_columns = [f'Area_{i + 1}' for i in range(20)]
        else:
            qor_columns = [f'Level_{i + 1}' for i in range(20)]

        try:
            qor_trajectory = torch.tensor(row[qor_columns].values.astype(np.float32), dtype=torch.float)
        except TypeError:
            qor_values = [float(val) for val in row[qor_columns].values]
            qor_trajectory = torch.tensor(qor_values, dtype=torch.float)

        # 从图中获取节点深度信息
        node_depths = graph.node_depth

        return {
            'design_name': design_name,
            'graph': graph,
            'recipe_indices': recipe_indices,
            'node_depths': node_depths,
            'qor_trajectory': qor_trajectory
        }

    def load_graph(self, design_name):
        """加载图文件并进行缓存"""
        if design_name in self.graph_cache:
            return self.graph_cache[design_name]

        graph_path = os.path.join(self.graph_dir, design_name.replace('.bench', '.pt'))

        if os.path.exists(graph_path):
            graph = torch.load(graph_path, map_location=torch.device('cpu'))
            self.graph_cache[design_name] = graph
            return graph
        else:
            raise FileNotFoundError(f"Graph file not found: {graph_path}")