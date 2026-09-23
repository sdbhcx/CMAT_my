"""
LAS training script with distributed support.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_model, get_loss_function, get_supported_models
from data.piadv2_dataset import get_dataloader as get_piadv2_dataloader
from data.piad_dataset import get_piad_dataloader
from data.laso_dataset import get_laso_dataloader
from utils.metrics import compute_metrics
from utils.utils import save_checkpoint, load_checkpoint, setup_logging

# print("--- Python 环境诊断 ---")
# print(f"Python 解释器路径: {sys.executable}")
# print(f"PyTorch 版本: {torch.__version__}")
# print(f"PyTorch 库文件位置: {torch.__file__}")
# print("------------------------")


def setup_distributed(rank, world_size):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def load_model_state_flexible(model, state_dict, strict=True, verbose=True):
    """加载模型权重。

    V1-A 新增了 description_selector 等模块，旧 checkpoint 里没有这些键。
    strict=False 时允许缺失键（新模块保持随机初始化），但会把缺失/多余键
    完整打印出来，避免"静默部分加载"这种不可审计的情况。

    Returns:
        missing_keys: 模型有、checkpoint 没有的键（新模块）
        unexpected_keys: checkpoint 有、模型没有的键（已废弃模块）
    """
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    if verbose and (missing or unexpected):
        print("[warm-start] 非严格加载：")
        if missing:
            print(f"  - 缺失键 {len(missing)} 个（随机初始化）: {missing[:8]}"
                  f"{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"  - 多余键 {len(unexpected)} 个（已忽略）: {unexpected[:8]}"
                  f"{' ...' if len(unexpected) > 8 else ''}")
    return missing, unexpected


def get_distributed_dataloader(config, split='train', rank=0, world_size=1):
    """Create distributed dataloader"""
    # Normalize dataset aliases.
    raw_type = str(config.get('dataset_type', 'piadv2')).lower()
    if raw_type in ('piad',):
        dataset_type = 'piad'
    elif raw_type in ('laso',):
        dataset_type = 'laso'
    elif raw_type in ('piadv2', 'piad_v2', 'piad2'):
        dataset_type = 'piadv2'
    else:
        dataset_type = 'piadv2'
    
    # 默认 None：非 piadv2 分支不会用到，避免出现未定义变量。
    description_bank = None
    piadv2_collate_fn = None

    if dataset_type == 'laso':
        from data.laso_dataset import LASODataset, collate_fn
        
        # Map split names for LASO (use the actual requested split)
        laso_split = 'test' if split == 'test' else split
        
        # Create LASO dataset
        dataset = LASODataset(
            run_type=laso_split,
            data_root=config['paths'].get('laso_data_root', None),
            num_points=config['data']['num_points'],
            use_augmentation=(split == 'train'),
            eval_setting='all'
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
        
        # Create dataloader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=(split == 'train'),
            collate_fn=collate_fn
        )
        
        return dataloader, sampler
        
    elif dataset_type == 'piad':
        from data.piad_dataset import PIADDataset
        
        # PIAD数据集路径
        data_root = config['paths']['data_root']
        if split == 'train':
            point_path = os.path.join(data_root, 'Point_Train.txt')
            img_path = os.path.join(data_root, 'Img_Train.txt')
            box_path = os.path.join(data_root, 'Box_Train.txt')
            use_augmentation = True
        else:
            point_path = os.path.join(data_root, 'Point_Test.txt')
            img_path = os.path.join(data_root, 'Img_Test.txt')
            box_path = os.path.join(data_root, 'Box_Test.txt')
            use_augmentation = False
        
        # Create PIAD dataset
        dataset = PIADDataset(
            run_type=split,
            setting_type=config.get('setting_type', 'Seen'),
            point_path=point_path,
            img_path=img_path,
            box_path=box_path,
            image_size=tuple(config['data']['image_size']),
            num_points=config['data']['num_points'],
            use_augmentation=use_augmentation,
            pair_num=config.get('pair_num', 2)
        )
    else:
        # PIADv2 visual-prompt dataset.
        from data.piadv2_dataset import PIADV2Dataset, load_description_bank_from_config, piadv2_collate_fn
        
        # 统一路径处理：优先使用小写文件名，兼容大写文件名
        data_root = config['paths']['data_root']

        # --- V1-A 固定描述池（未配置时为 None） ---
        description_bank = load_description_bank_from_config(config)
        
        if split == 'train':
            # 优先尝试小写文件名（PIADv2格式）
            point_path = os.path.join(data_root, 'Point_train.txt')
            img_path = os.path.join(data_root, 'Img_train.txt')
            # 如果小写文件不存在，尝试大写文件名（PIAD格式）
            if not os.path.exists(point_path):
                point_path = os.path.join(data_root, 'Point_Train.txt')
            if not os.path.exists(img_path):
                img_path = os.path.join(data_root, 'Img_Train.txt')
            use_augmentation = True
        else:
            # 优先尝试小写文件名（PIADv2格式）
            point_path = os.path.join(data_root, 'Point_test.txt')
            img_path = os.path.join(data_root, 'Img_test.txt')
            # 如果小写文件不存在，尝试大写文件名（PIAD格式）
            if not os.path.exists(point_path):
                point_path = os.path.join(data_root, 'Point_Test.txt')
            if not os.path.exists(img_path):
                img_path = os.path.join(data_root, 'Img_Test.txt')
            use_augmentation = False
        
        # Create dataset
        dataset = PIADV2Dataset(
            run_type=split,
            # 设定标签从 config 读取（Seen / Unseen_obj / Unseen_aff），
            # 不再硬编码，避免跨设定实验被静默标成 Seen。
            setting_type=config.get('setting_type', 'Seen'),
            point_path=point_path,
            img_path=img_path,
            image_size=config['data']['image_size'],
            num_points=config['data']['num_points'],
            use_augmentation=use_augmentation,
            text_source=config.get('data', {}).get('text_source', 'affordance_word'),
            description_bank=description_bank,
            description_count=config.get('data', {}).get('description_count', 4),
            on_bad_sample=config.get('data', {}).get('on_bad_sample', 'zeros'),
            # 问句查询（Question-as-Query）
            question_bank_path=config.get('data', {}).get('question_bank_path', None),
            question_missing=config.get('data', {}).get('question_missing', 'error')
        )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train')
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=piadv2_collate_fn if description_bank is not None else None
    )
    
    return dataloader, sampler


def _desc_attention_stats(alpha, eps=1e-8):
    """统计描述选择注意力 alpha [B,R,M] 的熵特征。

    归一化熵 = H(alpha) / log(有效描述条数)：
        ~1 表示接近均匀（没在做选择，等价于 mean）
        ~0 表示塌缩到单条描述（选择过于尖锐）
    只统计有效描述数 >= 2 的行；全无效或含 NaN 时返回 None。

    Returns:
        (归一化熵, max_alpha, 原始熵) 或 None
    """
    if alpha is None:
        return None
    a = alpha.detach().float()
    if a.dim() != 3 or a.numel() == 0:
        return None
    if not torch.isfinite(a).all():
        return None

    n_valid = (a > 1e-6).sum(dim=-1)                            # [B,R]
    sel = n_valid >= 2                                          # 至少 2 条才有"选择"可言
    if not bool(sel.any()):
        return None

    entropy = -(a * torch.log(a.clamp_min(eps))).sum(dim=-1)    # [B,R]
    ref = torch.log(n_valid[sel].to(a.dtype))                   # log(M_valid)

    ent_sel = entropy[sel]
    norm_ent = (ent_sel / ref.clamp_min(eps)).mean().item()
    max_alpha = a.max(dim=-1).values[sel].mean().item()
    raw_ent = ent_sel.mean().item()
    return norm_ent, max_alpha, raw_ent


class UnifiedTrainer:
    """
    Unified trainer class supporting both single and distributed training
    """
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        # Eval-time visual prompt ablation: 'none' | 'shuffle' | 'drop'
        self.ablate_prompt = str(config.get('ablate_prompt', 'none') or 'none').lower()

        # Setup device
        if self.is_distributed:
            self.device = torch.device(f'cuda:{rank}')
        else:
            # Check hardware config for device preference
            use_cuda = config.get('hardware', {}).get('use_cuda', True)
            if use_cuda and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        
        if rank == 0:
            print(f"Using device: {self.device}")
            if self.is_distributed:
                print(f"Distributed training on {world_size} GPUs")
        
        # Create unique experiment directory (only on rank 0)
        if rank == 0:
            name = self.config['name']
            model_name = self.config['model']['name']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 数据集名：优先用 dataset_type，缺失时回退到 data_root 的末级目录名
            dataset_name = self.config.get("dataset_type")
            if not dataset_name:
                dataset_name = os.path.basename(
                    os.path.normpath(self.config["paths"].get("data_root", "data")))
            dataset_name = str(dataset_name).lower().replace("-", "_").replace(" ", "_")

            # 数据集类型（seen / unseen_obj）：优先取 setting_type / eval_setting，
            # 缺失时从 data_root 路径里推断
            dataset_split = (self.config.get("setting_type")
                             or self.config.get("eval_setting")
                             or "").strip().lower()
            if not dataset_split:
                data_root = self.config["paths"].get("data_root", "")
                for tok in os.path.normpath(data_root).split(os.sep):
                    t = tok.lower()
                    if t.startswith("seen") or t.startswith("unseen"):
                        dataset_split = t
                        break
            if dataset_split == "unseen":
                # PIADv1 风格的裸 "Unseen" 归一为 unseen_obj；保留 unseen_obj / unseen_aff 原样
                dataset_split = "unseen_obj"

            prompt_type = str(self.config.get('model', {}).get('prompt_type', 'visual') or 'visual').lower()
            if self.config.get('model', {}).get('point_only', False):
                ablate_suffix = "_pointonly"
            elif prompt_type != 'visual':
                ablate_suffix = f"_{prompt_type}prompt"
            elif self.ablate_prompt != 'none':
                ablate_suffix = f"_ablate-{self.ablate_prompt}"
            else:
                ablate_suffix = ""
            if dataset_split:
                self.exp_name = f"{name}_{dataset_name}_{dataset_split}{ablate_suffix}_{timestamp}"
            else:
                self.exp_name = f"{name}_{dataset_name}{ablate_suffix}_{timestamp}"
            
            self.exp_dir = os.path.join(self.config['paths']['checkpoint_dir'], self.exp_name)
            self.log_dir = os.path.join(self.config['paths']['log_dir'], self.exp_name)
            
            os.makedirs(self.exp_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            
            print(f"Experiment directory: {self.exp_dir}")

        # Synchronize experiment path for all processes
        if self.is_distributed:
            # Broadcast exp_dir and log_dir to all processes
            dirs = [self.exp_dir if rank == 0 else None, self.log_dir if rank == 0 else None]
            dist.broadcast_object_list(dirs, src=0)
            if rank != 0:
                self.exp_dir, self.log_dir = dirs
        
        # Setup logging (only on rank 0)
        if rank == 0:
            self.setup_logging()
        
        # Create model
        model_name = config['model']['name']
        if rank == 0:
            print(f"Creating {model_name.upper()} model...")
        self.model = create_model(config).to(self.device)
        
        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)
        
        # Setup data loaders
        # Normalize dataset types.
        raw_type = str(config.get('dataset_type', 'piadv2')).lower()
        if raw_type in ('piad',):
            dataset_type = 'piad'
        elif raw_type in ('laso',):
            dataset_type = 'laso'
        elif raw_type in ('piadv2', 'piad_v2', 'piad2'):
            dataset_type = 'piadv2'
        else:
            dataset_type = 'piadv2'

        self.dataset_type = dataset_type
        
        if self.is_distributed:
            self.train_loader, self.train_sampler = get_distributed_dataloader(
                config, split='train', rank=rank, world_size=world_size
            )
            self.val_loader, self.val_sampler = get_distributed_dataloader(
                config, split='test', rank=rank, world_size=world_size
            )
            # For LASO in distributed mode, also prepare seen/unseen evaluation loaders (non-distributed)
            if dataset_type == 'laso':
                self.val_loader_seen = get_laso_dataloader(config, split='test', eval_setting='seen')
                self.val_loader_unseen = get_laso_dataloader(config, split='test', eval_setting='unseen')
        else:
            if dataset_type == 'laso':
                self.train_loader = get_laso_dataloader(config, split='train')
                self.val_loader = get_laso_dataloader(config, split='test')
                # Create additional seen/unseen test loaders for LASO
                self.val_loader_seen = get_laso_dataloader(config, split='test', eval_setting='seen')
                self.val_loader_unseen = get_laso_dataloader(config, split='test', eval_setting='unseen')
            elif dataset_type == 'piad':
                self.train_loader = get_piad_dataloader(config, split='train')
                self.val_loader = get_piad_dataloader(config, split='test')
                # For PIAD, we'll keep the existing seen/unseen structure
                self.val_loader_seen = None
                self.val_loader_unseen = None
            else:  # piadv2
                self.train_loader = get_piadv2_dataloader(config, split='train')
                self.val_loader = get_piadv2_dataloader(config, split='test')
                self.val_loader_seen = None
                self.val_loader_unseen = None
            self.train_sampler = None
            self.val_sampler = None
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup loss functions
        self.setup_losses()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_aiou = 0.0
        
        # V1-A 诊断信息：把"文本分支是否真的生效"在训练一开始就显式打出来，
        # 避免跑完几十个 epoch 才发现描述池没接上。
        if rank == 0:
            self._print_v1a_diagnostics()

        if rank == 0:
            print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
            print(f"Training dataset size: {len(self.train_loader.dataset)}")
            print(f"Validation dataset size: {len(self.val_loader.dataset)}")

    def _print_v1a_diagnostics(self):
        """打印文本分支 / 固定描述池 / 区域条件选择的实际生效状态。"""
        model_module = self.model.module if self.is_distributed else self.model
        data_cfg = self.config.get('data', {}) or {}
        desc_cfg = self.config.get('model', {}).get('description_selector', {}) or {}

        print("=" * 62)
        print("[V1-A 诊断] 文本分支 / 固定描述池 / 区域条件选择")
        print(f"  prompt_type            : {self.config.get('model', {}).get('prompt_type', 'N/A')}")
        print(f"  text_source            : {data_cfg.get('text_source', 'affordance_word')}")
        print(f"  description_bank_path  : {data_cfg.get('description_bank_path', None)}")
        print(f"  description_count (M)  : {data_cfg.get('description_count', 4)}")
        print(f"  description_selector   : mode={desc_cfg.get('mode', 'off')}, "
              f"hidden={desc_cfg.get('hidden_dim', 'N/A')}, "
              f"temp={desc_cfg.get('temperature', 'N/A')}")

        sel = getattr(model_module, 'description_selector', None)
        print(f"  模型内选择器已构建      : {sel is not None}")
        if sel is not None:
            n_trainable = sum(p.numel() for p in sel.parameters() if p.requires_grad)
            print(f"  选择器可训练参数        : {n_trainable:,}")
            print(f"  共享文本编码器          : "
                  f"{getattr(model_module, 'desc_text_encoder', None) is getattr(model_module, 'prompt_encoder', None)}")

        # 实际取一个 batch 验证描述字段是否真的被送进模型
        try:
            batch = next(iter(self.train_loader))
        except Exception as e:  # 取不到 batch 不影响启动，只记录
            print(f"  [WARNING] 无法取样例 batch 做端到端校验: {e}")
            print("=" * 62)
            return
        has_desc = 'descriptions' in batch
        print(f"  batch 含 'descriptions' : {has_desc}")
        if has_desc:
            rows = batch['descriptions']
            valid = batch.get('description_valid', None)
            print(f"  descriptions 形状       : B={len(rows)}, M={len(rows[0]) if rows else 0}")
            print(f"  description_valid 形状  : {tuple(valid.shape) if valid is not None else None}")
            print(f"  样例描述[0][0]          : {rows[0][0][:70]}")
            if valid is not None:
                print(f"  有效描述数/样本(均值)   : {valid.float().sum(1).mean().item():.2f}")
        print("=" * 62)
    
    def setup_logging(self):
        """Setup logging and tensorboard"""
        # Create log directory
        # The experiment directory is now created in __init__
        
        # Setup tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup file logging
        setup_logging(os.path.join(self.log_dir, 'training.log'))
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Base learning rate from config
        base_lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        # Ensure base_lr is a number
        if isinstance(base_lr, (list, tuple)):
            base_lr = base_lr[0]  # Take first value if it's a sequence
        elif isinstance(base_lr, str):
            base_lr = float(base_lr)  # Convert string to float (handles scientific notation)
        base_lr = float(base_lr)

        # Adjust base learning rate for distributed training if applicable
        lr = base_lr * self.world_size if self.is_distributed else base_lr
        
        model_module = self.model.module if self.is_distributed else self.model
        
        # LAS uses different learning rates for the prompt encoder and Point-MAE encoder.
        if self.config['model']['name'] == 'las':
            # Get prompt encoder parameters (either visual or text; absent for point-only models)
            if getattr(model_module, 'prompt_encoder', None) is not None:
                prompt_params = [p for p in model_module.prompt_encoder.parameters() if p.requires_grad]
            else:
                prompt_params = []
            prompt_params = list(prompt_params)

            # V1-A: 描述残差支路参数（选择器本体 + 可能的独立描述编码器）
            desc_selector_params = []
            desc_sel = getattr(model_module, 'description_selector', None)
            if desc_sel is not None:
                desc_selector_params = [p for p in desc_sel.parameters() if p.requires_grad]
                desc_enc = getattr(model_module, 'desc_text_encoder', None)
                # 与主 prompt 编码器不同对象时，视作文本编码器分支，沿用 prompt 的 LR
                if desc_enc is not None and desc_enc is not getattr(model_module, 'prompt_encoder', None):
                    prompt_params += [p for p in desc_enc.parameters() if p.requires_grad]
            
            # Group 2: Point-MAE parameters (LR * 0.2)
            pointmae_params = list(model_module.point_encoder.parameters())

            # Group 3: Rest of the model parameters (base LR)
            prompt_param_ids = {id(p) for p in prompt_params}
            pointmae_param_ids = {id(p) for p in pointmae_params}
            desc_param_ids = {id(p) for p in desc_selector_params}
            
            other_params = [
                p for p in model_module.parameters() 
                if id(p) not in prompt_param_ids
                and id(p) not in pointmae_param_ids
                and id(p) not in desc_param_ids
            ]

            # Determine prompt encoder type for naming
            prompt_type = self.config['model'].get('prompt_type', 'visual')
            prompt_name = f'{prompt_type}_encoder'

            desc_lr_scale = float(
                (self.config['model'].get('description_selector', {}) or {}).get('lr_scale', 1.0))

            param_groups = [
                {'params': prompt_params, 'lr': lr * 0.1, 'name': prompt_name},
                {'params': pointmae_params, 'lr': lr * 0.2, 'name': 'point_encoder'},
                {'params': desc_selector_params, 'lr': lr * desc_lr_scale,
                 'name': 'description_selector'},
                {'params': other_params, 'lr': lr, 'name': 'other_modules'}
            ]
            # drop empty groups (e.g. no prompt encoder in point-only mode)
            param_groups = [g for g in param_groups if g['params']]

            if self.rank == 0:
                print(f"Optimizer configured with parameter groups for LAS ({prompt_type} prompt):")
                total_params = 0
                for group in param_groups:
                    group_param_count = sum(p.numel() for p in group['params'])
                    total_params += group_param_count
                    print(f"  - Group '{group['name']}': {len(group['params'])} tensors, "
                          f"{group_param_count:,} parameters, lr={group['lr']:.2e}")
                print(f"Total trainable parameters: {total_params:,}")

            optimizer_type = self.config['training']['optimizer']
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
                
        else:
            # Default optimizer setup for other models
            if self.rank == 0:
                print(f"Using default optimizer for {self.config['model']['name']} model.")
            
            optimizer_type = self.config['training']['optimizer']
            if optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
    
    def setup_losses(self):
        """Setup loss functions"""
        self.loss_function = get_loss_function(self.config)
        self.model_name = self.config['model']['name'].lower()
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)
        
        # Set random seed for reproducible data loading in distributed training
        if self.is_distributed:
            torch.manual_seed(42 + self.epoch + self.rank)
            np.random.seed(42 + self.epoch + self.rank)
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0

        # ---- V1-A: 描述选择熵累加器 ----
        desc_stats_every = int(self.config.get('training', {}).get(
            'desc_attention_log_frequency',
            self.config.get('training', {}).get('print_frequency', 100)))
        desc_norm_ent_sum = 0.0
        desc_max_alpha_sum = 0.0
        desc_raw_ent_sum = 0.0
        desc_samples = 0
        # 在线方差：加权后描述嵌入在区域间的离散度（回答池化是否坍缩方差）
        desc_online_v_sum = 0.0
        desc_online_v_samples = 0

        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        else:
            progress_bar = self.train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Validate batch data consistency
            try:
                if 'points' in batch:
                    points_shape = batch['points'].shape
                    if len(points_shape) != 3 or points_shape[2] != 3:
                        print(f"Warning: Invalid points shape {points_shape}, skipping batch {batch_idx}")
                        continue
                    if points_shape[1] == 0:
                        print(f"Warning: Empty point cloud in batch {batch_idx}, skipping")
                        continue
                        
                if 'gt_mask' in batch:
                    mask_shape = batch['gt_mask'].shape
                    if 'points' in batch and mask_shape[1] != batch['points'].shape[1]:
                        print(f"Warning: Point-mask mismatch in batch {batch_idx}: {batch['points'].shape[1]} vs {mask_shape[1]}, skipping")
                        continue
                        
            except Exception as e:
                print(f"Error validating batch {batch_idx}: {e}, skipping")
                continue
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            loss, loss_dict = self.loss_function(
                outputs['segmentation_logits'],
                batch['gt_mask']
            )
            seg_loss = loss_dict['focal_loss']
            cont_loss = loss_dict['dice_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # ---- V1-A: 采样描述选择的熵（每 desc_stats_every 步一次，避免每步同步） ----
            if (self.rank == 0 and desc_stats_every > 0
                    and batch_idx % desc_stats_every == 0
                    and isinstance(outputs, dict)
                    and outputs.get('description_attention') is not None):
                _st = _desc_attention_stats(outputs['description_attention'])
                if _st is not None:
                    desc_norm_ent_sum += _st[0]
                    desc_max_alpha_sum += _st[1]
                    desc_raw_ent_sum += _st[2]
                    desc_samples += 1
                _ov = outputs.get('desc_online_variance')
                if _ov is not None:
                    desc_online_v_sum += float(_ov.detach().item()
                                               if hasattr(_ov, 'detach') else _ov)
                    desc_online_v_samples += 1

            # Update metrics
            total_loss += loss.item()
            seg_loss_total += seg_loss.item()
            cont_loss_total += cont_loss.item()
            
            # Update progress bar (only on rank 0)
            if self.rank == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Focal': f'{seg_loss.item():.4f}',
                    'Dice': f'{cont_loss.item():.4f}'
                })
                
                # Log to tensorboard
                global_step = self.epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/FocalLoss', seg_loss.item(), global_step)
                self.writer.add_scalar('Train/DiceLoss', cont_loss.item(), global_step)
        
        # Epoch averages
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_total / len(self.train_loader)
        avg_cont_loss = cont_loss_total / len(self.train_loader)

        # ---- V1-A: 输出描述选择的熵统计 ----
        if self.rank == 0 and desc_samples > 0:
            _ne = desc_norm_ent_sum / desc_samples
            _ma = desc_max_alpha_sum / desc_samples
            _re = desc_raw_ent_sum / desc_samples
            _gs = (self.epoch + 1) * len(self.train_loader)
            try:
                self.writer.add_scalar('Desc/norm_entropy', _ne, _gs)
                self.writer.add_scalar('Desc/max_alpha', _ma, _gs)
                self.writer.add_scalar('Desc/raw_entropy', _re, _gs)
            except Exception:
                pass
            _verdict = '均匀(未选择)' if _ne > 0.97 else ('塌缩' if _ne < 0.35 else '有区分度')
            print(f"  [V1-A] 描述选择: 归一化熵={_ne:.4f} (1=均匀, 0=塌缩到单条) "
                  f"max_alpha={_ma:.4f} 原始熵={_re:.4f} | 采样{desc_samples}次 -> {_verdict}")
        elif self.rank == 0 and desc_samples == 0:
            print("  [V1-A] 描述选择: 本 epoch 未采到 description_attention"
                  f"（mode={self.config.get('model', {}).get('description_selector', {}).get('mode', 'n/a')}）")

        if self.rank == 0 and desc_online_v_samples > 0:
            _ov = desc_online_v_sum / desc_online_v_samples
            try:
                self.writer.add_scalar('Desc/online_V', _ov,
                                       (self.epoch + 1) * len(self.train_loader))
            except Exception:
                pass
            print(f"  [V1-A] 加权后描述嵌入区域间方差={_ov:.4f} "
                  f"(mean 模式理论为 0；显著大于 0 说明条件加权保住了多样性)")

        return avg_loss, avg_seg_loss, avg_cont_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Only show progress bar on rank 0
            if self.rank == 0:
                val_iterator = tqdm(self.val_loader, desc="Validating")
            else:
                val_iterator = self.val_loader
                
            for batch in val_iterator:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # V1 ablation: feed mismatched images to destroy image<->point
                # correspondence while keeping token count and feature statistics
                if self.ablate_prompt == 'shuffle' and isinstance(batch.get('image'), torch.Tensor):
                    perm = torch.randperm(batch['image'].size(0), device=batch['image'].device)
                    batch['image'] = batch['image'][perm]

                # Forward pass
                outputs = self.model(batch)
                
                loss, loss_dict = self.loss_function(
                    outputs['segmentation_logits'],
                    batch['gt_mask']
                )
                seg_loss = loss_dict['focal_loss']
                cont_loss = loss_dict['dice_loss']
                
                # Update metrics
                total_loss += loss.item()
                seg_loss_total += seg_loss.item()
                cont_loss_total += cont_loss.item()
                
                # Collect predictions for metrics
                predictions = torch.sigmoid(outputs['segmentation_logits']).cpu().numpy()
                targets = batch['gt_mask'].cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Compute metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Epoch averages
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = seg_loss_total / len(self.val_loader)
        avg_cont_loss = cont_loss_total / len(self.val_loader)
        
        return avg_loss, avg_seg_loss, avg_cont_loss, metrics
    
    def validate_seen_unseen(self):
        """Validate the model on seen/unseen splits for LASO dataset"""
        if self.dataset_type != 'laso':
            return {}
        
        results = {}
        
        # Evaluate on seen split
        if hasattr(self, 'val_loader_seen') and self.val_loader_seen is not None:
            print(f"\n[Rank {self.rank}] Evaluating on SEEN split...")
            seen_results = self._evaluate_split(self.val_loader_seen, "Seen")
            for key, value in seen_results.items():
                results[f'seen_{key}'] = value
        
        # Evaluate on unseen split  
        if hasattr(self, 'val_loader_unseen') and self.val_loader_unseen is not None:
            print(f"\n[Rank {self.rank}] Evaluating on UNSEEN split...")
            unseen_results = self._evaluate_split(self.val_loader_unseen, "Unseen")
            for key, value in unseen_results.items():
                results[f'unseen_{key}'] = value
        
        return results
    
    def _evaluate_split(self, dataloader, split_name):
        """Evaluate model on a specific data split"""
        self.model.eval()
        
        total_loss = 0
        seg_loss_total = 0
        cont_loss_total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            # Only show progress bar on rank 0
            if self.rank == 0:
                iterator = tqdm(dataloader, desc=f"Evaluating {split_name}")
            else:
                iterator = dataloader
                
            for batch in iterator:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                loss, loss_dict = self.loss_function(
                    outputs['segmentation_logits'],
                    batch['gt_mask']
                )
                seg_loss = loss_dict['focal_loss']
                cont_loss = loss_dict['dice_loss']
                
                total_loss += loss.item()
                seg_loss_total += seg_loss.item()
                cont_loss_total += cont_loss.item()
                
                # Collect predictions and targets for metrics
                pred = torch.sigmoid(outputs['segmentation_logits'])
                all_predictions.append(pred.cpu())
                all_targets.append(batch['gt_mask'].cpu())
        
        # Compute metrics (use numpy for consistency with main validate)
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate detailed metrics using the same function as main validation
        metrics = compute_metrics(all_predictions, all_targets)
        
        # Epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_seg_loss = seg_loss_total / len(dataloader)
        avg_cont_loss = cont_loss_total / len(dataloader)
        
        return {
            'loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'cont_loss': avg_cont_loss,
            'aiou': metrics['aiou'],
            'auc': metrics['auc'],
            'sim': metrics['sim'],
            'mae': metrics['mae']
        }
    
    def run_eval_only(self):
        """Run a single evaluation pass (for prompt-ablation studies on existing checkpoints)."""
        logging.info(f"Eval-only mode (ablate_prompt={self.ablate_prompt})")
        val_loss, val_seg_loss, val_cont_loss, val_metrics = self.validate()
        if self.rank == 0:
            logging.info(f"[eval-only] ablate_prompt={self.ablate_prompt}")
            logging.info(f"  Val Loss: {val_loss:.4f} (Focal: {val_seg_loss:.4f}, Dice: {val_cont_loss:.4f})")
            logging.info(f"  Val aIoU: {val_metrics['aiou']:.4f}")
            logging.info(f"  Val AUC: {val_metrics['auc']:.4f}")
            logging.info(f"  Val SIM: {val_metrics['sim']:.4f}")
            logging.info(f"  Val MAE: {val_metrics['mae']:.4f}")
            if self.dataset_type == 'laso':
                seen_unseen_results = self.validate_seen_unseen()
                logging.info(f"  Seen/Unseen: {seen_unseen_results}")
        print("Evaluation completed!")

    def train(self):
        """Main training loop"""
        if self.config.get('eval_only', False):
            self.run_eval_only()
            return

        logging.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Training
            train_loss, train_seg_loss, train_cont_loss = self.train_epoch()
            
            # Validation
            val_loss, val_seg_loss, val_cont_loss, val_metrics = self.validate()
            
            # LASO Seen/Unseen evaluation (every 5 epochs to avoid too much overhead)
            seen_unseen_results = {}
            if self.dataset_type == 'laso':
                seen_unseen_results = self.validate_seen_unseen()
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Logging (only on rank 0)
            if self.rank == 0:
                logging.info(f"Epoch {epoch}:")
                logging.info(f"  Train Loss: {train_loss:.4f} (Focal: {train_seg_loss:.4f}, Dice: {train_cont_loss:.4f})")
                logging.info(f"  Val Loss: {val_loss:.4f} (Focal: {val_seg_loss:.4f}, Dice: {val_cont_loss:.4f})")
                logging.info(f"  Val aIoU: {val_metrics['aiou']:.4f}")
                logging.info(f"  Val AUC: {val_metrics['auc']:.4f}")
                logging.info(f"  Val SIM: {val_metrics['sim']:.4f}")
                logging.info(f"  Val MAE: {val_metrics['mae']:.4f}")

                # Log seen/unseen results if available
                if seen_unseen_results:
                    logging.info(f"  --- LASO Seen/Unseen Results ---")
                    if 'seen_aiou' in seen_unseen_results:
                        logging.info(
                            f"  Seen aIoU: {seen_unseen_results['seen_aiou']:.4f}, "
                            f"AUC: {seen_unseen_results['seen_auc']:.4f}, "
                            f"SIM: {seen_unseen_results.get('seen_sim', float('nan')):.4f}, "
                            f"MAE: {seen_unseen_results.get('seen_mae', float('nan')):.4f}"
                        )
                    if 'unseen_aiou' in seen_unseen_results:
                        logging.info(
                            f"  Unseen aIoU: {seen_unseen_results['unseen_aiou']:.4f}, "
                            f"AUC: {seen_unseen_results['unseen_auc']:.4f}, "
                            f"SIM: {seen_unseen_results.get('unseen_sim', float('nan')):.4f}, "
                            f"MAE: {seen_unseen_results.get('unseen_mae', float('nan')):.4f}"
                        )
                
                # Tensorboard logging
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
                
                self.writer.add_scalar('Train/EpochFocalLoss', train_seg_loss, epoch)
                self.writer.add_scalar('Train/EpochDiceLoss', train_cont_loss, epoch)
                self.writer.add_scalar('Val/EpochFocalLoss', val_seg_loss, epoch)
                self.writer.add_scalar('Val/EpochDiceLoss', val_cont_loss, epoch)
                
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Save checkpoint
            is_best = val_metrics['aiou'] > self.best_val_aiou
            if is_best:
                self.best_val_aiou = val_metrics['aiou']
                self.best_val_loss = val_loss
            
            # Use the experiment-specific directory for saving checkpoints
            checkpoint_path = os.path.join(self.exp_dir, 'checkpoint.pth')

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_aiou': self.best_val_aiou,
                'best_val_loss': self.best_val_loss,
                'config': self.config
            }, is_best, checkpoint_path)
            
            if self.rank == 0:
                print(f"  Best Val aIoU: {self.best_val_aiou:.4f}")
                print("-" * 50)
        
        print("Training completed!")
        if self.rank == 0:
            self.writer.close()

def train_worker(rank, world_size, config, resume_path=None):
    """Training worker function for distributed training"""
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    try:
        # Create trainer
        trainer = UnifiedTrainer(config, rank=rank, world_size=world_size)
        
        # Resume from checkpoint if specified
        if resume_path:
            checkpoint = load_checkpoint(resume_path)
            
            # Handle DDP state dict
            state_dict = checkpoint['model_state_dict']
            if not list(state_dict.keys())[0].startswith('module.') and world_size > 1:
                # Add 'module.' prefix for DDP
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif list(state_dict.keys())[0].startswith('module.') and world_size == 1:
                # Remove 'module.' prefix for single GPU
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # warm-start: 旧 checkpoint 缺少新增模块（如 description_selector）时允许部分加载
            warm_start = bool(config.get('training', {}).get('warm_start', False))
            missing, unexpected = load_model_state_flexible(
                trainer.model, state_dict, strict=not warm_start, verbose=(rank == 0))

            if warm_start and missing:
                # 新增模块是随机初始化的，优化器/调度器状态维度已不匹配，必须丢弃。
                if rank == 0:
                    print("[warm-start] 检测到新增模块，跳过 optimizer/scheduler 状态恢复，"
                          "epoch 从 checkpoint 继续但优化器从头开始。")
            else:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if trainer.scheduler and checkpoint['scheduler_state_dict']:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_aiou = checkpoint.get('best_val_aiou', checkpoint.get('best_val_iou', 0.0))
            trainer.best_val_loss = checkpoint['best_val_loss']
            
            if rank == 0:
                print(f"Resumed from epoch {trainer.epoch}")

        # Start training (or single eval pass)
        if config.get('eval_only', False):
            trainer.run_eval_only()
        else:
            trainer.train()

    finally:
        # Clean up distributed training
        cleanup_distributed()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='LAS training script')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of GPUs for distributed training')
    parser.add_argument('--ablate-prompt', type=str, default='none',
                       choices=['none', 'shuffle', 'drop'],
                       help='Eval-time visual prompt ablation: shuffle=mismatched images, '
                            'drop=point-only forward (prompt tokens removed)')
    parser.add_argument('--eval-only', action='store_true',
                       help='Skip training; run one validation pass and exit '
                            '(use with --resume on an existing checkpoint)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['ablate_prompt'] = args.ablate_prompt
    config['eval_only'] = args.eval_only
    
    # Validate model type
    supported_models = get_supported_models()
    model_name = config['model']['name'].lower()
    if model_name not in supported_models:
        raise ValueError(f"Model '{model_name}' not supported. "
                        f"Supported models: {supported_models}")
    
    print(f"Training {model_name.upper()} model with config: {args.config}")
    
    # Create base directories, experiment-specific ones are created in the trainer
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Check for distributed training
    if args.distributed or args.world_size > 1:
        # Distributed training
        world_size = args.world_size
        print(f"Starting distributed training on {world_size} GPUs")
        
        # Spawn training processes
        mp.spawn(
            train_worker,
            args=(world_size, config, args.resume),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        print("Starting single GPU training")
        trainer = UnifiedTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint = load_checkpoint(args.resume)
            warm_start = bool(config.get('training', {}).get('warm_start', False))
            missing, unexpected = load_model_state_flexible(
                trainer.model, checkpoint['model_state_dict'],
                strict=not warm_start, verbose=True)
            if warm_start and missing:
                print("[warm-start] 检测到新增模块，跳过 optimizer/scheduler 状态恢复。")
            else:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if trainer.scheduler and checkpoint['scheduler_state_dict']:
                    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_aiou = checkpoint.get('best_val_aiou', checkpoint.get('best_val_iou', 0.0))
            trainer.best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from epoch {trainer.epoch}")

        # Start training (or single eval pass)
        if config.get('eval_only', False):
            trainer.run_eval_only()
        else:
            trainer.train()

if __name__ == '__main__':
    main()