"""
PIADv2 dataset implementation for LAS visual-prompt training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
import random
import os

def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

# PIADv2 affordance categories (24 classes).
# 提到模块级：描述池加载时可直接用它做覆盖校验，无需先实例化数据集。
PIADV2_AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab', 'carry', 'ride', 'clean',
    'play', 'beat', 'speak', 'pull'
]


class PIADV2Dataset(Dataset):
    """
    Dataset class for LAS training on PIADv2
    Returns a dictionary containing:
    - 'image': preprocessed image tensor
    - 'points': original point cloud tensor (N x 3)
    - 'gt_mask': pixel-level ground truth mask (N x 1)
    - 'affordance_id': affordance category ID (0-23)
    - 'instance_id': unique ID for different 3D models
    """
    
    def __init__(self,
                 run_type='train',
                 setting_type='Seen',
                 point_path=None,
                 img_path=None,
                 image_size=(224, 224),
                 num_points=2048,
                 use_augmentation=True,
                 data_root=None,
                 text_source='affordance_word',
                 description_bank=None,
                 description_count=4,
                 on_bad_sample='zeros',
                 question_bank_path=None,
                 question_missing='error'):

        super().__init__()

        self.run_type = run_type
        self.setting_type = setting_type
        self.image_size = image_size
        self.num_points = num_points
        self.use_augmentation = use_augmentation
        self.text_source = text_source
        # --- V1-A 固定描述池 ---
        # description_bank 为 None 时不产出描述字段，行为与改动前完全一致。
        self.description_bank = description_bank
        self.description_count = int(description_count or 4)
        if on_bad_sample not in ('zeros', 'error'):
            raise ValueError(f"on_bad_sample 必须是 'zeros' 或 'error'，收到 {on_bad_sample!r}")
        self.on_bad_sample = on_bad_sample
        # 注：描述池覆盖率统计在 affordance_label_list 定义之后打印（见下方）。
        
        # PIADv2 affordance categories (24 classes)
        self.affordance_label_list = list(PIADV2_AFFORDANCE_LABELS)
        if self.description_bank is not None:
            covered = sum(1 for a in self.affordance_label_list if a in self.description_bank)
            missing = [a for a in self.affordance_label_list if a not in self.description_bank]
            print(f"[PIADV2Dataset] 描述池覆盖 {covered}/{len(self.affordance_label_list)} 个功能类别, "
                  f"M={self.description_count}")
            if missing:
                print(f"[PIADV2Dataset] WARNING: 描述池缺失类别（将用占位文本填充）: {missing}")
        
        # Load file paths
        self.img_files = self._read_file_list(img_path)
        self.point_files = self._read_file_list(point_path)

        # Load natural-language text prompts (hk/ok) if requested;
        # default 'affordance_word' uses bare affordance label (no file needed).
        self.text_list = None
        if self.text_source in ('hk', 'ok') and data_root is not None:
            split_suffix = 'train' if run_type == 'train' else 'test'
            text_file = os.path.join(data_root, f'{self.text_source}_{split_suffix}.txt')
            print(f"[PIADV2Dataset] Loading text prompts from: {text_file}")
            with open(text_file, 'r', encoding='utf-8') as f:
                self.text_list = [line.rstrip('\n') for line in f if line.strip()]
            if len(self.text_list) != len(self.img_files):
                print(f"[PIADV2Dataset] WARNING: text list length {len(self.text_list)} != "
                      f"img list length {len(self.img_files)}")
            print(f"[PIADV2Dataset] Sample text: {self.text_list[0][:120]}")

        # Debug: print a few resolved paths to help diagnose path issues
        if len(self.img_files) > 0:
            print("[PIADV2Dataset] Sample image path resolved:", self.img_files[0])
        if len(self.point_files) > 0:
            print("[PIADV2Dataset] Sample point path resolved:", self.point_files[0])
        
        if self.run_type == 'train':
            self.object_point_map = self._create_object_point_map()
        
        # Create instance mappings
        self.instance_mapping = self._create_instance_mapping()
        
        # Image preprocessing
        self.image_transform = self._get_image_transform()

        # --- 问句查询（Question-as-Query）---
        # text_source='question' 时按 (Object, Affordance) 查问句表：
        # train 随机取 Question1..N（强制对措辞鲁棒），val/test 固定 Question0。
        if question_missing not in ('error', 'fallback'):
            raise ValueError(f"question_missing 必须是 'error' 或 'fallback'，收到 {question_missing!r}")
        self.question_missing = question_missing
        self.question_bank = {}
        if question_bank_path is not None:
            self.question_bank = self._load_question_bank(question_bank_path)
            print(f"[PIADV2Dataset] 问句表载入 {len(self.question_bank)} 个 (Object, Affordance) 组合: "
                  f"{question_bank_path}")

    def _load_question_bank(self, path):
        """读问句 CSV -> {(Object, Affordance): [Question0..QuestionN]}。"""
        import csv as _csv

        if not os.path.isabs(path):
            # 相对路径按项目根解析，与描述池的行为保持一致
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(project_root, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"问句表文件不存在: {path}")
        bank = {}
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = _csv.DictReader(f)
            qcols = [c for c in (reader.fieldnames or []) if c.startswith('Question')]
            if not qcols:
                raise ValueError(f"问句表缺少 Question* 列: {path}")

            def _qidx(col):
                tail = col[len('Question'):]
                return int(tail) if tail.isdigit() else 10 ** 6

            qcols.sort(key=_qidx)
            for row in reader:
                key = (row['Object'], row['Affordance'])
                qs = [(row[c] or '').strip() for c in qcols]
                if not any(qs):
                    raise ValueError(f"问句表组合 {key} 全为空: {path}")
                bank[key] = qs
        if not bank:
            raise ValueError(f"问句表为空: {path}")
        return bank

    def _find_question_text(self, object_name, affordance):
        """返回 (question_text, question_id)。缺项按 question_missing 处理。"""
        qs = self.question_bank.get((object_name, affordance))
        if qs is None or not any(qs):
            msg = f"问句表缺少组合 ({object_name}, {affordance})"
            if self.question_missing == 'error':
                raise KeyError(msg)
            print(f"[PIADV2Dataset] WARNING: {msg}，回退到 affordance 词")
            return affordance, -1
        # Question0 规范问句留给 val/test；train 随机取一条改写，避免模型记模板
        qid = 0 if self.run_type != 'train' else random.randint(1, len(qs) - 1)
        return qs[qid], qid
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        """
        Returns a dictionary with keys:
        - 'image': preprocessed image tensor
        - 'points': point cloud tensor (N x 3)
        - 'gt_mask': ground truth mask (N x 1)
        - 'affordance_id': affordance category ID
        - 'instance_id': unique instance ID
        """
        img_path = self.img_files[index]

        # 物体类别与 affordance 均从路径解析：.../ObjectClass/Instance/Affordance/xxx
        object_name = img_path.split('/')[-4]
        img_affordance_name = img_path.split('/')[-2]

        if self.run_type == 'train':
            # Dynamic sampling for training to ensure correct image-point cloud pairs.
            # This logic is inspired by the PIAD dataset to prevent mismatches.
            candidate_indices = self.object_point_map.get(object_name)
            if not candidate_indices:
                # Handle cases where an object in the image list has no corresponding point cloud
                raise ValueError(f"No point clouds found for object category: {object_name}")

            while True:
                point_idx = random.choice(candidate_indices)
                point_path = self.point_files[point_idx]
                pc_affordance_name = point_path.split('/')[-2]
                if img_affordance_name == pc_affordance_name:
                    break
        else:
            # Static mapping for validation/testing, assuming files are perfectly aligned
            point_path = self.point_files[index]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size)
        image = self.image_transform(image)
        
        # Load and preprocess point cloud
        points, gt_mask, point_ids = self._load_point_cloud(point_path)
        points = self._preprocess_points(points)
        
        # Get affordance and instance IDs
        affordance_id = self._get_affordance_id(img_path)
        instance_id = self._get_instance_id(img_path)

        if self.text_source == 'question':
            if not self.question_bank:
                raise RuntimeError("text_source='question' 但未载入问句表（question_bank_path 未配置）")
            text, question_id = self._find_question_text(object_name, img_affordance_name)
        else:
            text = (self.text_list[index] if self.text_list is not None
                    else self.affordance_label_list[affordance_id])
            question_id = None

        sample = {
            'image': image,
            'points': torch.from_numpy(points).float(),
            'gt_mask': torch.from_numpy(gt_mask).float(),
            'affordance_id': affordance_id,
            'instance_id': instance_id,
            # text prompt: question (问句查询) / hk-ok 自然语言 / bare affordance word
            'text': text,
            'object_name': object_name,
            'question_id': -1 if question_id is None else question_id,
        }

        # --- V1-A 固定描述池字段 ---
        # 点采样同步作用于 point_ids；重复采样允许重复索引值。
        if self.description_bank is not None:
            query_key = self.affordance_label_list[affordance_id]
            descriptions, valid = self.description_bank.get_padded(
                query_key, self.description_count)
            if not any(valid):
                # 未知查询且无描述：显式退回原查询路径，不静默补入其他动作的描述。
                print(f"[PIADV2Dataset] WARNING: 描述池缺少 query_key '{query_key}'，"
                      f"该样本描述残差为零（回退到原查询路径）")
            sample['query_key'] = query_key
            sample['descriptions'] = descriptions
            sample['description_valid'] = torch.tensor(valid, dtype=torch.bool)
            sample['point_ids'] = torch.from_numpy(point_ids).long()

        return sample
    
    def _read_file_list(self, path):
        """Read file list from text file with improved path resolution"""
        if path is None:
            return []
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        fixed_files = []
        missing_samples = 0
        max_warn = 10
        
        print(f"[PIADV2Dataset] Reading file list from: {path}")
        
        with open(path, 'r') as f:
            for line_num, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                    
                original_line = line
                resolved_path = None
                
                # 如果是绝对路径，直接使用
                if os.path.isabs(line):
                    resolved_path = line
                else:
                    # 相对路径解析策略
                    # 1. 首先尝试相对于项目根目录
                    candidate1 = os.path.join(project_root, line)
                    if os.path.exists(candidate1):
                        resolved_path = candidate1
                    else:
                        # 2. 尝试路径修正策略
                        # 处理 Data/Seen/ -> Data/PIADv2/Seen/ 的情况
                        if line.startswith('Data/Seen/'):
                            alt_line = 'Data/PIADv2/Seen/' + line[len('Data/Seen/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 处理 Data/Unseen/ -> Data/PIADv2/Unseen/ 的情况
                        elif line.startswith('Data/Unseen/'):
                            alt_line = 'Data/PIADv2/Unseen/' + line[len('Data/Unseen/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 处理 Data/Unseen_obj/ -> Data/PIADv2/Unseen_obj/ 的情况
                        elif line.startswith('Data/Unseen_obj/'):
                            alt_line = 'Data/PIADv2/Unseen_obj/' + line[len('Data/Unseen_obj/'):]
                            candidate2 = os.path.join(project_root, alt_line)
                            if os.path.exists(candidate2):
                                resolved_path = candidate2
                        
                        # 3. 尝试相对于 txt 文件所在目录解析
                        if resolved_path is None:
                            list_dir = os.path.dirname(path)
                            # 去掉 "Data/" 前缀，从 list_dir 的父目录解析
                            # txt 在 data_root/ 下，路径如 Data/Unseen_obj/... → 去掉 Data/ 后从 data_root 的父目录拼接
                            if line.startswith('Data/'):
                                parent_dir = os.path.dirname(list_dir)
                                alt = os.path.join(parent_dir, line[len('Data/'):])
                            else:
                                alt = os.path.join(list_dir, line)
                            if os.path.exists(alt):
                                resolved_path = alt

                        # 4. 尝试相对于 txt 文件所在目录解析（完整路径）
                        if resolved_path is None:
                            alt2 = os.path.join(list_dir, line)
                            if os.path.exists(alt2):
                                resolved_path = alt2

                        # 如果所有策略都失败，使用原始路径（让错误暴露出来）
                        if resolved_path is None:
                            resolved_path = candidate1
                
                # 转换为绝对路径
                abs_path = os.path.abspath(resolved_path)
                
                # 检查文件是否存在
                if not os.path.exists(abs_path):
                    if missing_samples < max_warn:
                        print(f"[PIADV2Dataset] Warning: File not found at line {line_num}")
                        print(f"  Original: {original_line}")
                        print(f"  Resolved: {abs_path}")
                    missing_samples += 1
                
                fixed_files.append(abs_path)
        
        if missing_samples > 0:
            print(f"[PIADV2Dataset] Total missing files: {missing_samples}/{len(fixed_files)}")
            if missing_samples > max_warn:
                print(f"[PIADV2Dataset] (Only showing first {max_warn} warnings)")
        
        return fixed_files
    
    def _load_point_cloud(self, path):
        """Load point cloud and ground truth mask"""
        try:
            data = np.load(path)
            points = data[:, :3]  # xyz coordinates
            gt_mask = data[:, 3:]  # ground truth mask
            
            # Validate data dimensions
            if points.shape[0] == 0:
                raise ValueError(f"Empty point cloud in {path}")
            if points.shape[1] != 3:
                raise ValueError(f"Invalid point cloud format in {path}: expected 3 coordinates, got {points.shape[1]}")
            if gt_mask.shape[0] != points.shape[0]:
                raise ValueError(f"Point-mask dimension mismatch in {path}: {points.shape[0]} vs {gt_mask.shape[0]}")
            
            # Sample points if necessary
            if len(points) > self.num_points:
                # Use fixed random seed for reproducible sampling during validation
                if self.run_type != 'train':
                    np.random.seed(hash(path) % 2**32)
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
                gt_mask = gt_mask[indices]
            elif len(points) < self.num_points:
                # Ensure we have at least one point to duplicate
                if len(points) == 0:
                    raise ValueError(f"Cannot sample from empty point cloud in {path}")
                
                # Pad with duplicated points
                diff = self.num_points - len(points)
                # Use modulo to safely handle small point clouds
                padded = np.random.choice(len(points), diff, replace=True)
                indices = np.concatenate([np.arange(len(points)), padded])
                points = points[indices]
                gt_mask = gt_mask[indices]
            else:
                indices = np.arange(len(points))
            indices = np.asarray(indices, dtype=np.int64)
            
            # Final validation
            assert points.shape[0] == self.num_points, f"Final point count mismatch: {points.shape[0]} != {self.num_points}"
            assert gt_mask.shape[0] == self.num_points, f"Final mask count mismatch: {gt_mask.shape[0]} != {self.num_points}"
            assert indices.shape[0] == self.num_points, f"Final index count mismatch: {indices.shape[0]} != {self.num_points}"
            
            return points, gt_mask, indices
            
        except Exception as e:
            message = f"Error loading point cloud from {path}: {e}"
            if self.on_bad_sample == 'error':
                # 禁止把全零兜底样本当作真实标签：直接暴露，交由调用方修复数据。
                raise RuntimeError(message) from e
            print(message)
            # Return a fallback point cloud with zeros
            fallback_points = np.zeros((self.num_points, 3), dtype=np.float32)
            fallback_mask = np.zeros((self.num_points, 1), dtype=np.float32)
            return fallback_points, fallback_mask, np.zeros(self.num_points, dtype=np.int64)
    
    def _preprocess_points(self, points):
        """Normalize and augment point cloud"""
        # Normalize to unit sphere
        points, _, _ = pc_normalize(points)
        
        # Apply augmentation if training
        if self.use_augmentation and self.run_type == 'train':
            # Add jitter
            points += np.random.normal(0, 0.01, points.shape)
            
            # Random rotation around z-axis
            if np.random.rand() > 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([
                    [cos_angle, -sin_angle, 0],
                    [sin_angle, cos_angle, 0],
                    [0, 0, 1]
                ])
                points = np.dot(points, rotation_matrix.T)
        
        return points
    
    def _get_image_transform(self):
        """Get image preprocessing transform"""
        if self.run_type == 'train' and self.use_augmentation:
            return transforms.Compose([
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_affordance_id(self, img_path):
        """Extract affordance ID from image path"""
        affordance_name = img_path.split('/')[-2]
        return self.affordance_label_list.index(affordance_name)
    
    def _get_instance_id(self, img_path):
        """Extract instance ID from image path"""
        # Extract object name and instance info from path
        parts = img_path.split('/')
        object_name = parts[-4]
        instance_info = parts[-3]
        
        # Create unique instance ID
        instance_key = f"{object_name}_{instance_info}"
        return self.instance_mapping.get(instance_key, 0)
    
    def _create_instance_mapping(self):
        """Create mapping from instance strings to unique IDs"""
        instance_set = set()
        
        # Collect all unique instances
        for img_path in self.img_files:
            parts = img_path.split('/')
            object_name = parts[-4]
            instance_info = parts[-3]
            instance_key = f"{object_name}_{instance_info}"
            instance_set.add(instance_key)
        
        # Create mapping
        return {instance: idx for idx, instance in enumerate(sorted(instance_set))}

    def _create_object_point_map(self):
        """
        Creates a map from an object category to a list of its point cloud indices.
        This is used during training for robust data sampling, ensuring that an image
        is paired with a point cloud of the same object category and affordance.
        Path format is assumed to be .../ObjectClass/Instance/Affordance/xxx.npy
        """
        object_map = {}
        for i, p_path in enumerate(self.point_files):
            try:
                object_name = p_path.split('/')[-4]
                if object_name not in object_map:
                    object_map[object_name] = []
                object_map[object_name].append(i)
            except IndexError:
                # This may happen if a path in the list does not follow the expected format.
                print(f"Warning: Could not parse object name from path: {p_path}. Skipping this entry.")
        return object_map

def piadv2_collate_fn(batch):
    """显式 collate：保证 descriptions 保持 B×M，不被默认 collate 转置。

    默认 collate 遇到 list 元素会做 zip(*batch)，把 B 份 M 条描述变成 M 份 B 条，
    这会破坏 batch 维度与 description_valid 的对应关系，因此这里逐个字段显式处理。
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(first, str):
            collated[key] = list(values)          # 不转置
        elif isinstance(first, list):
            collated[key] = [list(v) for v in values]   # 保持 B×M，不转置
        elif isinstance(first, (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values
    return collated


def load_description_bank_from_config(config, required_keys=None, verbose=True):
    """按配置加载固定描述池；未配置时返回 None（保持旧行为）。"""
    data_cfg = config.get('data', {}) or {}
    bank_path = data_cfg.get('description_bank_path', None)
    if not bank_path:
        return None

    from data.description_bank import DescriptionBank
    # 未显式指定时，按 PIADv2 的 24 个功能类别做覆盖校验：
    # 缺类别会直接报错，而不是静默退化为「零残差」。
    if required_keys is None:
        required_keys = PIADV2_AFFORDANCE_LABELS
    return DescriptionBank.load(
        path=bank_path,
        description_count=int(data_cfg.get('description_count', 4)),
        missing=data_cfg.get('missing_description', 'error'),
        required_keys=required_keys,
    )


def get_dataloader(config, split='train', affordance_label_list=None):
    """Create dataloader for specified split"""
    
    # 统一路径处理：优先使用小写文件名，兼容大写文件名
    data_root = config['paths']['data_root']
    
    def get_file_path(base_name, split_suffix):
        """获取文件路径，优先小写，兼容大写"""
        # 优先尝试小写文件名（PIADv2格式）
        lowercase_path = os.path.join(data_root, f'{base_name}_{split_suffix.lower()}.txt')
        if os.path.exists(lowercase_path):
            return lowercase_path
        
        # 尝试大写文件名（PIAD格式）
        uppercase_path = os.path.join(data_root, f'{base_name}_{split_suffix.capitalize()}.txt')
        if os.path.exists(uppercase_path):
            return uppercase_path
        
        # 如果都不存在，返回小写路径（让后续错误处理机制处理）
        return lowercase_path
    
    # Determine paths based on split
    if split == 'train':
        point_path = get_file_path('Point', 'train')
        img_path = get_file_path('Img', 'train')
        use_augmentation = True
    elif split == 'val':
        point_path = get_file_path('Point', 'val')
        img_path = get_file_path('Img', 'val')
        use_augmentation = False
    else:  # test
        point_path = get_file_path('Point', 'test')
        img_path = get_file_path('Img', 'test')
        use_augmentation = False
    
    # 固定描述池（未配置时为 None，数据集不产出描述字段）
    description_bank = load_description_bank_from_config(
        config, required_keys=affordance_label_list)

    # Create dataset
    dataset = PIADV2Dataset(
        run_type=split,
        # 设定标签从 config 读取（Seen / Unseen_obj / Unseen_aff）
        setting_type=config.get('setting_type', 'Seen'),
        point_path=point_path,
        img_path=img_path,
        image_size=config['data']['image_size'],
        num_points=config['data']['num_points'],
        use_augmentation=use_augmentation,
        data_root=data_root,
        text_source=config.get('data', {}).get('text_source', 'affordance_word'),
        description_bank=description_bank,
        description_count=config.get('data', {}).get('description_count', 4),
        on_bad_sample=config.get('data', {}).get('on_bad_sample', 'zeros'),
        # 问句查询（Question-as-Query）
        question_bank_path=config.get('data', {}).get('question_bank_path', None),
        question_missing=config.get('data', {}).get('question_missing', 'error')
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(split == 'train'),
        # 描述字段需要显式 collate，否则默认 collate 会转置 B×M
        collate_fn=piadv2_collate_fn if description_bank is not None else None
    )
    
    return dataloader