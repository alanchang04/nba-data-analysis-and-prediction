import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime
from tqdm import tqdm
import os
import math
import warnings

# 抑制警告訊息
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.basicConfig(level=logging.ERROR)

class ExponentialGating(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.scale * x + self.bias)

class ImprovedMatrixMemory(nn.Module):
    def __init__(self, dim: int, blocksize: int):
        super().__init__()
        self.blocksize = blocksize
        self.num_blocks = dim // blocksize
        self.memory = nn.Parameter(
            torch.randn(self.num_blocks, blocksize, blocksize) / math.sqrt(blocksize)
        )
        self.scale = nn.Parameter(torch.ones(1, 1, self.num_blocks))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, self.num_blocks, self.blocksize)
        x = torch.einsum('blmd,mde->blme', x, self.memory)
        x = x * self.scale.unsqueeze(-1)
        return x.reshape(B, L, D)

# 資料類別定義
@dataclass
class mLSTMLayerConfig:
    """mLSTM 層配置"""
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4

@dataclass
class mLSTMBlockConfig:
    """mLSTM 區塊配置"""
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

@dataclass
class FeedForwardConfig:
    """前饋網路配置"""
    proj_factor: float = 1.3
    act_fn: str = "gelu"

@dataclass
class sLSTMLayerConfig:
    """sLSTM 層配置"""
    backend: str = "cuda"
    num_heads: int = 4
    conv1d_kernel_size: int = 4
    bias_init: str = "powerlaw_blockdependent"

@dataclass
class sLSTMBlockConfig:
    """sLSTM 區塊配置"""
    slstm: sLSTMLayerConfig = field(default_factory=sLSTMLayerConfig)
    feedforward: FeedForwardConfig = field(default_factory=FeedForwardConfig)

@dataclass
class xLSTMBlockStackConfig:
    """xLSTM 堆疊配置"""
    mlstm_block: mLSTMBlockConfig = field(default_factory=mLSTMBlockConfig)
    slstm_block: sLSTMBlockConfig = field(default_factory=sLSTMBlockConfig)
    context_length: int = 256
    num_blocks: int = 7
    embedding_dim: int = 128
    slstm_at: List[int] = field(default_factory=list)

def convert_metrics_for_json(metrics):
    """將指標轉換為可序列化的 JSON 格式"""
    converted_metrics = []
    for metric in metrics:
        converted_metric = {
            'loss': float(metric['loss']),
            'accuracy': float(metric['accuracy']),
            'predictions': metric['predictions'].tolist() if hasattr(metric['predictions'], 'tolist') else metric['predictions'],
            'targets': metric['targets'].tolist() if hasattr(metric['targets'], 'tolist') else metric['targets']
        }
        converted_metrics.append(converted_metric)
    return converted_metrics

class FeatureProcessor:
    """特徵處理器：負責資料前處理和特徵工程"""
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """建立和轉換特徵

        Args:
            df: 包含原始比賽資料的 DataFrame

        Returns:
            df_new: 包含所有處理後特徵的新 DataFrame
        """
        print("正在建立特徵...")
        df_new = df.copy()
        df_new = df_new.sort_values(['Season', 'Date'])
        
        for team in df_new['home_team_name'].unique():
            # 處理主場數據
            home_mask = df_new['home_team_name'] == team
            df_new.loc[home_mask, 'home_last_5_wins'] = (
                df_new.loc[home_mask, 'Result']
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
            
            df_new.loc[home_mask, 'home_season_wr'] = (
                df_new.loc[home_mask, 'Result']
                .shift(1)
                .expanding()
                .mean()
            )
            
            df_new.loc[home_mask, 'home_streak'] = (
                df_new.loc[home_mask, 'Result']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .sum()
            )
            
            # 處理客場數據
            guest_mask = df_new['guest_team_name'] == team
            df_new.loc[guest_mask, 'guest_last_5_wins'] = (
                df_new.loc[guest_mask, 'Result']
                .shift(1)
                .rolling(window=5, min_periods=1)
                .mean()
            )
            
            df_new.loc[guest_mask, 'guest_season_wr'] = (
                df_new.loc[guest_mask, 'Result']
                .shift(1)
                .expanding()
                .mean()
            )
            
            df_new.loc[guest_mask, 'guest_streak'] = (
                df_new.loc[guest_mask, 'Result']
                .shift(1)
                .rolling(window=3, min_periods=1)
                .sum()
            )
        
        # 計算休息天數
        df_new['home_rest_days'] = df_new.groupby('home_team_name')['Date'].diff().dt.days
        df_new['guest_rest_days'] = df_new.groupby('guest_team_name')['Date'].diff().dt.days
        
        # 計算賽季進度
        df_new['season_progress'] = df_new.groupby('Season').cumcount() / \
                                   df_new.groupby('Season').size()
        
        # 填補缺失值
        df_new = df_new.fillna(0)
        return df_new
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple:
        """準備機器學習用的特徵和目標變數

        Args:
            df: 包含特徵的 DataFrame
            feature_cols: 要使用的特徵欄位名稱列表
            target_col: 目標變數欄位名稱

        Returns:
            features: 標準化後的特徵張量
            targets: 目標變數張量
        """
        # 提取特徵和目標
        features = df[feature_cols].values
        targets = df[target_col].values
        
        # 標準化特徵
        features = self.scaler.fit_transform(features)
        
        # 轉換為 PyTorch 張量
        return torch.FloatTensor(features), torch.FloatTensor(targets)

class BasketballDataset(Dataset):
    """籃球資料集類別"""
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    def __init__(self, config, dim: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        return x

class Conv1dLayer(nn.Module):
    """一維卷積層"""
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :-self.conv.padding[0]]
        x = self.activation(x)
        return x.transpose(1, 2)

class MatrixMemory(nn.Module):
    """矩陣記憶模組"""
    def __init__(self, dim: int, blocksize: int):
        super().__init__()
        self.blocksize = blocksize
        self.memory = nn.Parameter(torch.randn(dim // blocksize, blocksize, blocksize))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x = x.view(B, L, -1, self.blocksize)
        x = torch.einsum('blmd,mde->blme', x, self.memory)
        return x.reshape(B, L, D)

class mLSTMLayer(nn.Module):
    """mLSTM 層"""
    def __init__(self, config: mLSTMLayerConfig, dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(config, dim)
        self.conv = Conv1dLayer(dim, config.conv1d_kernel_size)
        self.matrix_memory = ImprovedMatrixMemory(dim, config.qkv_proj_blocksize)
        
        # 使用指數門控
        self.input_gate = ExponentialGating(dim)
        self.forget_gate = ExponentialGating(dim)
        self.output_gate = ExponentialGating(dim)
        self.memory_gate = nn.Linear(dim * 2, dim)
        
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple:
        if state is None:
            state = torch.zeros_like(x)
        
        # 注意力和記憶處理
        attended = self.attention(x)
        convolved = self.conv(x)
        memory = self.matrix_memory(x)
        
        combined = self.layer_norm1(attended + convolved + memory)
        combined = self.dropout(combined)
        
        # 狀態更新
        concat = torch.cat([combined, state], dim=-1)
        i = self.input_gate(combined)
        f = self.forget_gate(state)
        o = self.output_gate(combined)
        m = torch.tanh(self.memory_gate(concat))
        
        new_state = f * state + i * m
        output = o * torch.tanh(new_state)
        output = self.layer_norm2(output + x)
        
        return output, new_state

class sLSTMLayer(nn.Module):
    """sLSTM 層"""
    def __init__(self, config: sLSTMLayerConfig, dim: int):
        super().__init__()
        self.attention = MultiHeadAttention(config, dim)
        self.conv = Conv1dLayer(dim, config.conv1d_kernel_size)
        
        # 使用指數門控
        self.state_gate = ExponentialGating(dim)
        self.update_gate = ExponentialGating(dim)
        self.transform = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
        if config.bias_init == "powerlaw_blockdependent":
            self._init_powerlaw_bias()
            
    def _init_powerlaw_bias(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.1)
                    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple:
        if state is None:
            state = torch.zeros_like(x)
            
        # 注意力處理
        attended = self.attention(x)
        convolved = self.conv(x)
        combined = self.layer_norm1(attended + convolved)
        combined = self.dropout(combined)
        
        # 狀態更新
        s = self.state_gate(combined)
        u = self.update_gate(combined)
        
        transformed = self.transform(combined)
        new_state = s * state + (1 - s) * transformed
        
        output = u * new_state + (1 - u) * x
        output = self.layer_norm2(output)
        
        return output, new_state

class FeedForward(nn.Module):
    """前饋網路"""
    def __init__(self, config: FeedForwardConfig, dim: int):
        super().__init__()
        hidden_dim = int(dim * config.proj_factor)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU() if config.act_fn == "gelu" else nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.net(x))

class xLSTMBlock(nn.Module):
    """xLSTM 區塊"""
    def __init__(self, config: xLSTMBlockStackConfig, block_type: str):
        super().__init__()
        dim = config.embedding_dim
        
        if block_type == 'mlstm':
            self.lstm = mLSTMLayer(config.mlstm_block.mlstm, dim)
        else:  # slstm
            self.lstm = sLSTMLayer(config.slstm_block.slstm, dim)
            self.ff = FeedForward(config.slstm_block.feedforward, dim)
            
        self.block_type = block_type
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple:
        output, new_state = self.lstm(x, state)
        if self.block_type == 'slstm':
            output = self.ff(output)
        return output, new_state

class XLSTMPredictor(nn.Module):
    """xLSTM 預測器"""
    def __init__(self, input_size: int, hidden_size: int = 512, num_blocks: int = 4):
        super().__init__()
        config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(),
            slstm_block=sLSTMBlockConfig(),
            context_length=1,
            num_blocks=num_blocks,
            embedding_dim=hidden_size,
            slstm_at=[1]
        )
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.xlstm = xLSTMBlockStack(config)
        
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.input_proj(x)
        x, _ = self.xlstm(x)
        x = x.mean(dim=1)
        
        return self.output(x).squeeze(-1)

class TransformerPredictor(nn.Module):
    """Transformer 預測器"""
    def __init__(self, input_size: int, hidden_size: int = 512, n_head: int = 16, n_layers: int = 8):
        super().__init__()
        
        self.input_dropout = nn.Dropout(0.1)
        self.embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 新增: 位置編碼
        self.position_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.feature_extraction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(4)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size*6,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False
        )
        
        self.global_context = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_dropout(x)
        x = self.embedding(x.unsqueeze(1))
        x = x + self.position_embedding
        
        multi_scale_features = []
        for feature_extractor in self.feature_extraction:
            multi_scale_features.append(feature_extractor(x))
        x = torch.mean(torch.stack(multi_scale_features), dim=0)
        
        transformer_out = self.transformer(x)
        context_out, _ = self.global_context(transformer_out, transformer_out, transformer_out)
        
        combined_features = torch.cat([transformer_out, context_out], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        pooled_features = torch.mean(fused_features, dim=1)
        
        return self.output(pooled_features).squeeze(-1)

class xLSTMBlockStack(nn.Module):
    """xLSTM 區塊堆疊"""
    def __init__(self, config: xLSTMBlockStackConfig):
        super().__init__()
        self.config = config
        
        if config.slstm_at is None:
            config.slstm_at = []
            
        self.blocks = nn.ModuleList([
            xLSTMBlock(config, 'slstm' if i in config.slstm_at else 'mlstm')
            for i in range(config.num_blocks)
        ])
    
    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> tuple:
        if states is None:
            states = [None] * len(self.blocks)
            
        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block(x, state)
            new_states.append(new_state)
            
        return x, new_states

class ModelTrainer:
    """模型訓練器"""
    def __init__(self, model_type: str, config: dict, device: str):
        self.model_type = model_type
        self.config = config
        self.device = device
        self.criterion = nn.BCELoss()
        self.models = self._create_models()
        self.optimizers = [
            optim.AdamW(model.parameters(), 
                      lr=config['learning_rate'],
                      weight_decay=0.01)
            for model in self.models
        ]
        self.schedulers = [
            optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10, 
                T_mult=2
            )
            for optimizer in self.optimizers
        ]
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def _load_best_models(self):
        """載入最佳模型狀態"""
        for model, optimizer, best_state in zip(
            self.models,
            self.optimizers,
            self.best_model_states
        ):
            model.load_state_dict(best_state['model_state'])
            optimizer.load_state_dict(best_state['optimizer_state'])

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float):
        """儲存檢查點"""
        save_dir = f'models/{self.model_type}/checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        info_path = os.path.join(save_dir, 'checkpoint_info.json')

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                existing_info = json.load(f)
                existing_acc = existing_info.get('val_accuracy', 0.0)
                
                if existing_acc > val_acc:
                    print(f"檢查點未更新 - 目前準確率 ({val_acc:.4f}) 低於現有準確率 ({existing_acc:.4f})")
                    return

        save_info = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'config': self.config
        }

        for i, model in enumerate(self.models):
            save_path = os.path.join(save_dir, f'best_checkpoint_{i}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[i].state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'config': self.config
            }, save_path)

        with open(info_path, 'w') as f:
            json.dump(save_info, f, indent=4)

        print(f"\n已更新檢查點 - 新的最佳準確率: {val_acc:.4f}")

    def save_final_best(self):
        """儲存最終最佳模型"""
        save_dir = f'models/{self.model_type}/best_model'
        os.makedirs(save_dir, exist_ok=True)
        info_path = os.path.join(save_dir, 'final_model_info.json')

        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                existing_info = json.load(f)
                existing_acc = existing_info.get('val_accuracy', 0.0)
                
                if existing_acc > self.best_val_acc:
                    print(f"\n最終模型未更新 - 目前準確率 ({self.best_val_acc:.4f}) 低於現有準確率 ({existing_acc:.4f})")
                    return

        save_info = {
            'val_loss': float(self.best_val_loss),
            'val_accuracy': float(self.best_val_acc),
            'config': self.config
        }

        for i, model in enumerate(self.models):
            save_path = os.path.join(save_dir, f'final_best_model_{i}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': self.best_val_loss,
                'val_accuracy': self.best_val_acc,
                'config': self.config
            }, save_path)

        with open(info_path, 'w') as f:
            json.dump(save_info, f, indent=4)

        print(f"\n已更新最終最佳模型 - 最佳準確率: {self.best_val_acc:.4f}")

    def _create_models(self) -> List[nn.Module]:
        """建立模型"""
        input_size = len(self.config['feature_columns'])
        params = self.config['model_params']
        model_num = params.get('model_num', 3)
        
        if self.model_type == 'transformer':
            return [
                TransformerPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    n_head=params['n_head'],
                    n_layers=params['n_layers']
                ).to(self.device)
                for _ in range(model_num)
            ]
        else:  # xlstm
            return [
                XLSTMPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_blocks=params['n_layers']
                ).to(self.device)
                for _ in range(model_num)
            ]
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Tuple[List[float], List[Dict]]:
        """訓練模型"""
        train_losses = []
        val_metrics = []
        
        progress_bar = tqdm(range(self.config['num_epochs']), 
                          desc=f'訓練 {self.model_type}')
        batch_progress = tqdm(total=len(train_loader), desc='當前輪次', leave=False)
        
        try:
            for epoch in progress_bar:
                batch_progress.reset()
                batch_progress.set_description(f'輪次 {epoch+1}')
                
                train_loss = self._train_epoch(train_loader, batch_progress)
                train_losses.append(train_loss)
                
                # 更新學習率
                for scheduler in self.schedulers:
                    scheduler.step()
                
                if val_loader:
                    val_results = self.evaluate(val_loader)
                    val_metrics.append(val_results)
                    
                    status = f'損失: {train_loss:.4f}, 驗證損失: {val_results["loss"]:.4f}'
                    status += f', 驗證準確率: {val_results["accuracy"]:.4f}'
                    
                    if val_results['accuracy'] > self.best_val_acc:
                        self.best_val_loss = val_results['loss']
                        self.best_val_acc = val_results['accuracy']
                        status += ' ✓'
                        
                        self.save_checkpoint(
                            epoch=epoch,
                            val_loss=val_results['loss'],
                            val_acc=val_results['accuracy']
                        )
                else:
                    status = f'損失: {train_loss:.4f}'
                
                progress_bar.set_postfix_str(status)
                
        except Exception as e:
            print(f"\n訓練中斷: {str(e)}")
            if hasattr(self, 'best_model_states'):
                self.save_final_best()
            raise e
        
        finally:
            progress_bar.close()
            batch_progress.close()
            
            if val_loader and hasattr(self, 'best_model_states'):
                self._load_best_models()
                self.save_final_best()
        
        return train_losses, val_metrics

    def _train_epoch(self, train_loader: DataLoader, batch_progress: tqdm) -> float:
        """訓練一個輪次"""
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            batch_loss = 0
            for model, optimizer in zip(self.models, self.optimizers):
                model.train()
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                batch_loss += loss.item()
            
            total_loss += batch_loss / len(self.models)
            batch_progress.update(1)
            
            # 更新進度條資訊
            batch_progress.set_postfix({
                'loss': f'{batch_loss/len(self.models):.4f}'
            })
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """評估模型"""
        all_preds = []
        all_targets = []
        total_loss = 0
        
        eval_progress = tqdm(test_loader, desc='驗證中', leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in eval_progress:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_preds = []
                for model in self.models:
                    model.eval()
                    pred = model(batch_X)
                    batch_preds.append(pred)
                
                # 模型集成
                batch_pred = torch.stack(batch_preds).mean(dim=0)
                loss = self.criterion(batch_pred, batch_y)
                
                all_preds.extend(batch_pred.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                total_loss += loss.item()
                
                current_acc = accuracy_score(
                    (np.array(all_targets) > 0.5).astype(int),
                    (np.array(all_preds) > 0.5).astype(int)
                )
                
                eval_progress.set_postfix({
                    'loss': f'{total_loss/len(all_preds):.4f}',
                    'acc': f'{current_acc:.4f}'
                })
        
        eval_progress.close()
        
        final_preds = np.array(all_preds)
        final_targets = np.array(all_targets)
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy_score(
                (final_targets > 0.5).astype(int),
                (final_preds > 0.5).astype(int)
            ),
            'predictions': final_preds,
            'targets': final_targets
        }

# 圖片標題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字型
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

def plot_comparison(transformer_metrics, xlstm_metrics, save_path=None):
    """繪製比較圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot([x['loss'] for x in transformer_metrics], 
             label='Transformer', color='blue', alpha=0.7)
    ax1.plot([x['loss'] for x in xlstm_metrics], 
             label='XLSTM', color='red', alpha=0.7)
    ax1.set_title('驗證損失比較')
    ax1.set_xlabel('訓練輪次')
    ax1.set_ylabel('損失值')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot([x['accuracy'] for x in transformer_metrics], 
             label='Transformer', color='blue', alpha=0.7)
    ax2.plot([x['accuracy'] for x in xlstm_metrics], 
             label='XLSTM', color='red', alpha=0.7)
    ax2.set_title('驗證準確率比較')
    ax2.set_xlabel('訓練輪次')
    ax2.set_ylabel('準確率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函式執行流程"""
    print("開始 NBA 比賽預測系統比較...")
    
    # 設置運算裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用裝置: {device}")
    if device.type == 'cuda':
        print(f"GPU型號: {torch.cuda.get_device_name(0)}")
        print(f"可用 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
        print(f"目前 GPU 記憶體使用量: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    # 載入和預處理數據
    print("\n載入資料中...")
    df = pd.read_excel('regular_season.xlsx')
    print(f"已載入 {len(df)} 場比賽資料")
    
    # 分割數據
    print("分割訓練集和測試集...")
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    split_date = pd.to_datetime('2021-05-16')
    train_data = df[df['Date'] < split_date]
    test_data = df[df['Date'] >= split_date]
    print(f"訓練集大小: {len(train_data)}, 測試集大小: {len(test_data)}")

    # 模型配置
    config = {
        'feature_columns': [
            'home_last_5_wins', 'guest_last_5_wins',
            'home_season_wr', 'guest_season_wr',
            'home_rest_days', 'guest_rest_days',
            'season_progress', 'home_streak', 'guest_streak'
        ],
        'target_column': 'Result',
        'learning_rate': 0.0003,
        'batch_size': 64,
        'num_epochs': 300,
        'model_params': {
            'hidden_size': 512,
            'n_head': 16,
            'n_layers': 8,
            'model_num': 5
        }
    }

    # 準備特徵處理器
    feature_processor = FeatureProcessor()
    
    # 處理訓練數據
    print("\n準備訓練資料...")
    train_processed = feature_processor.create_features(train_data)
    X_train, y_train = feature_processor.prepare_data(
        train_processed,
        config['feature_columns'],
        config['target_column']
    )

    # 處理測試數據
    print("\n準備測試資料...")
    test_processed = feature_processor.create_features(test_data)
    X_test, y_test = feature_processor.prepare_data(
        test_processed,
        config['feature_columns'],
        config['target_column']
    )
    
    # 建立資料載入器
    train_dataset = BasketballDataset(X_train, y_train)
    test_dataset = BasketballDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # # 訓練和評估 Transformer 模型
    # print("\n開始訓練 Transformer 模型...")
    # transformer_trainer = ModelTrainer('transformer', config, device)
    # transformer_losses, transformer_metrics = transformer_trainer.train(train_loader, test_loader)
    
    # 訓練和評估 XLSTM 模型
    print("\n開始訓練 XLSTM 模型...")
    xlstm_trainer = ModelTrainer('xlstm', config, device)
    xlstm_losses, xlstm_metrics = xlstm_trainer.train(train_loader, test_loader)
    
    # 最終評估結果
    print("\n最終評估結果:")
    # print("\nTransformer 模型:")
    # transformer_final = transformer_trainer.evaluate(test_loader)
    # print(f"測試準確率: {transformer_final['accuracy']:.4f}")
    # print("\n分類報告:")
    # print(classification_report(
    #     transformer_final['targets'],
    #     (transformer_final['predictions'] > 0.5).astype(int)
    # ))
    
    print("\nXLSTM 模型:")
    xlstm_final = xlstm_trainer.evaluate(test_loader)
    print(f"測試準確率: {xlstm_final['accuracy']:.4f}")
    print("\n分類報告:")
    print(classification_report(
        xlstm_final['targets'],
        (xlstm_final['predictions'] > 0.5).astype(int)
    ))
    
    # # 繪製比較圖表
    # print("\n繪製比較結果...")
    # plot_comparison(transformer_metrics, xlstm_metrics, 'model_comparison.png')
    
    # # 儲存比較結果
    # comparison_results = {
    #     'transformer': {
    #         'final_accuracy': float(transformer_final['accuracy']),
    #         'validation_metrics': convert_metrics_for_json(transformer_metrics)
    #     },
    #     'xlstm': {
    #         'final_accuracy': float(xlstm_final['accuracy']),
    #         'validation_metrics': convert_metrics_for_json(xlstm_metrics)
    #     }
    # }
    
    # 建立結果儲存目錄
    os.makedirs('results', exist_ok=True)
    
    # # 儲存結果
    # with open('results/comparison_results.json', 'w') as f:
    #     json.dump(comparison_results, f, indent=4)
    
    print("\n比較完成！結果已儲存至 'results' 目錄。")

if __name__ == "__main__":
    main()