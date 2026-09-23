"""
区域条件化描述选择（region-conditional description selection）V1-A。

设计依据（docs/Affordance_CMAT_Architecture_Pipeline_Losses.md 11.4）：

    U = W_u LN(G)                       [B,R,d]
    V = W_v E_desc                      [B,M,d]
    alpha = softmax_m(U V^T / sqrt(d))  [B,R,M]
    D_r   = sum_m alpha_rm V_m          [B,R,d]
    delta = W_out GELU(W_f [U; D; U*D]) [B,R,D]

    G' = G + delta   ->  原 feature_propagation -> 原 co-attention -> 原分割头

三种模式：
- off        : 旧路径，模块不参与计算（调用方通常也不会构造/调用）。
- mean       : 对有效描述均匀加权，保留相同的投影与融合结构（B1）。
- conditional: 使用区域相关的 alpha（B2）。

数值约束：
- padding / 无效描述用有限的大负数屏蔽，**禁止对全 -inf 做 softmax**。
- 整行无有效描述时 alpha 与 delta 直接置零，实现「未知查询退回原查询路径」。
- 只将 W_out 的权重和 bias 零初始化，使首次前向等价于原模型；
  不同时添加零初始化乘法门，避免双零阻断学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_MODES = ('off', 'mean', 'conditional')

# 用于屏蔽无效描述列的有限大负数（避免 -inf 参与 softmax 产生 NaN）
_MASK_NEG = -1e4


class ConditionalTextSelector(nn.Module):
    """在区域 token 上按几何条件选择功能描述，并输出残差 delta。"""

    def __init__(self, geom_dim, text_dim, hidden_dim=256, mode='conditional',
                 zero_init_output=True, temperature=1.0, mask_neg=_MASK_NEG):
        """
        Args:
            geom_dim: 区域特征维度 D（等于 unified_dim）
            text_dim: 描述编码维度 E（roberta-base 为 768）
            hidden_dim: 选择空间维度 d
            mode: 'off' | 'mean' | 'conditional'
            zero_init_output: W_out 权重与 bias 置零
            temperature: softmax 温度，>1 更平滑，<1 更尖锐
            mask_neg: 屏蔽无效描述用的有限大负数
        """
        super().__init__()

        mode = str(mode or 'off').lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"不支持的 description selector 模式: {mode!r}，可选 {SUPPORTED_MODES}")

        self.geom_dim = int(geom_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.zero_init_output = bool(zero_init_output)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError(f"temperature 必须为正数，收到 {self.temperature}")
        self.mask_neg = float(mask_neg)

        self.geom_norm = nn.LayerNorm(self.geom_dim)
        self.geom_proj = nn.Linear(self.geom_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)

        # [U; D; U*D] -> d -> D
        self.fuse = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.geom_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.geom_proj.weight)
        nn.init.zeros_(self.geom_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

        if self.zero_init_output:
            # 只零初始化输出层：首步 delta == 0，前向与原模型一致。
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------ 前向

    def compute_attention(self, geom_features, desc_emb, desc_valid=None):
        """计算 alpha [B,R,M]（已屏蔽无效描述，无有效描述的行为 0）。"""
        u = self.geom_proj(self.geom_norm(geom_features))            # [B,R,d]
        v = self.text_proj(desc_emb)                                 # [B,M,d]

        scale = self.hidden_dim ** 0.5
        scores = torch.matmul(u, v.transpose(-1, -2)) / scale        # [B,R,M]
        if self.temperature != 1.0:
            scores = scores / self.temperature

        b, m = desc_emb.shape[0], desc_emb.shape[1]
        if desc_valid is None:
            valid = torch.ones(b, m, dtype=torch.bool, device=scores.device)
        else:
            valid = desc_valid.to(dtype=torch.bool, device=scores.device)
            if valid.shape != (b, m):
                raise ValueError(
                    f"description_valid 形状应为 {(b, m)}，收到 {tuple(valid.shape)}")

        # 每行是否至少有一条有效描述
        row_valid = valid.any(dim=1, keepdim=True)                   # [B,1]
        safe_valid = valid | (~row_valid)                            # 全无效行临时全 True，避免 NaN

        num_regions = geom_features.shape[1]
        if self.mode == 'mean':
            # 均匀加权：与 conditional 共用同一套投影/融合结构，仅权重计算不同。
            # 两个必须注意的点：
            #   1) 按有效条数归一化，否则 M 条描述直接累加、幅值随 M 线性增长；
            #   2) 先在 [B,M] 上算好，再显式 expand 到 [B,R,M]——
            #      直接拿 2 维 weights 与 [B,1,M] 相乘会被广播成 [B,B,M]。
            w = safe_valid.to(scores.dtype)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = w.unsqueeze(1).expand(b, num_regions, m)
        else:
            scores = scores.masked_fill(~safe_valid.unsqueeze(1), self.mask_neg)
            weights = torch.softmax(scores, dim=-1)

        weights = weights * safe_valid.unsqueeze(1).to(weights.dtype)
        weights = weights * row_valid.unsqueeze(-1).to(weights.dtype)  # 全无效行整体置零
        return weights

    def forward(self, geom_features, desc_emb, desc_valid=None):
        """
        Args:
            geom_features: [B,R,D] 区域特征（point_projection 之后）
            desc_emb:      [B,M,E] 描述编码（masked mean pooling 结果）
            desc_valid:    [B,M] bool，padding 或未知查询为 False

        Returns:
            delta:  [B,R,D] 残差，加到 geom_features 上
            alpha:  [B,R,M] 选择权重（诊断用）
        """
        if geom_features.dim() != 3:
            raise ValueError(f"geom_features 应为 [B,R,D]，收到 {tuple(geom_features.shape)}")
        if desc_emb.dim() != 3:
            raise ValueError(f"desc_emb 应为 [B,M,E]，收到 {tuple(desc_emb.shape)}")
        if desc_emb.size(1) == 0:
            raise ValueError("desc_emb 的 M 维为 0；描述数必须 >= 1")

        if self.mode == 'off':
            delta = torch.zeros_like(geom_features)
            alpha = torch.zeros(geom_features.shape[0], geom_features.shape[1],
                                desc_emb.shape[1],
                                device=geom_features.device, dtype=geom_features.dtype)
            return delta, alpha

        alpha = self.compute_attention(geom_features, desc_emb, desc_valid)

        u = self.geom_proj(self.geom_norm(geom_features))            # [B,R,d]
        v = self.text_proj(desc_emb)                                 # [B,M,d]
        aggregated = torch.matmul(alpha, v)                          # [B,R,d]

        fused = torch.cat([u, aggregated, u * aggregated], dim=-1)   # [B,R,3d]
        delta = self.out_proj(F.gelu(self.fuse(fused)))              # [B,R,D]

        # 整行无有效描述时，delta 也必须为零。
        # 注意不能只靠 aggregated=0：fused 里还有 u 项，out_proj 会把它变成非零残差。
        if desc_valid is None:
            row_valid = torch.ones(delta.shape[0], 1, dtype=torch.bool,
                                   device=delta.device)
        else:
            row_valid = desc_valid.to(dtype=torch.bool).any(dim=1, keepdim=True)
        delta = delta * row_valid.unsqueeze(-1).to(delta.dtype)

        return delta, alpha

    # ------------------------------------------------------------------ 诊断

    @staticmethod
    def attention_entropy(alpha, desc_valid=None, eps=1e-8):
        """alpha 的香农熵 [B,R]（只统计有效列），用于监测选择是否塌缩。"""
        a = alpha.clamp_min(eps)
        if desc_valid is not None:
            a = a * desc_valid.unsqueeze(1).to(a.dtype)
        log_a = torch.log(a.clamp_min(eps))
        return -(a * log_a).sum(dim=-1)

    def extra_repr(self):
        return (f"geom_dim={self.geom_dim}, text_dim={self.text_dim}, "
                f"hidden_dim={self.hidden_dim}, mode={self.mode}, "
                f"zero_init_output={self.zero_init_output}")


def build_description_selector(geom_dim, text_dim, cfg=None):
    """按配置构造选择器；cfg 为 None 或 mode=off 时返回 None。"""
    if cfg is None:
        return None
    mode = str(cfg.get('mode', 'off') or 'off').lower()
    if mode == 'off':
        return None
    return ConditionalTextSelector(
        geom_dim=geom_dim,
        text_dim=text_dim,
        hidden_dim=int(cfg.get('hidden_dim', 256)),
        mode=mode,
        zero_init_output=bool(cfg.get('zero_init_output', True)),
        temperature=float(cfg.get('temperature', 1.0)),
    )
