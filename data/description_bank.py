"""
固定功能描述池（fixed description bank）V1-A。

设计依据（docs/Affordance_CMAT_Architecture_Pipeline_Losses.md 11.3）：
- 描述池是**固定大小**的候选解释集合，按 query_key（功能动作）索引。
- 描述覆盖不同的可成立机制，而不是同义改写。
- 描述不读取实例 mask 或测试答案，也不包含对当前实例部件存在性的断言。
- 缺失条目必须在启动时报告，禁止静默补入其他动作的描述。
- 银行文件带版本与哈希，写入日志以区分实验。

本模块只负责加载、去重、完整性检查与查表，不做任何编码。
"""

import hashlib
import json
import os

DEFAULT_DESCRIPTION_COUNT = 4

_VALID_MISSING_POLICIES = ('error', 'report')


class DescriptionBankError(RuntimeError):
    """描述池文件结构或完整性不满足要求。"""


class DescriptionBank:
    """按 query_key 索引的固定描述池。

    entries: dict[query_key] -> list[str]，每个列表长度固定为 description_count。
    """

    def __init__(self, entries, version=None, language=None, generator=None,
                 prompt_version=None, description_count=DEFAULT_DESCRIPTION_COUNT,
                 source_path=None, metadata=None, views=None, view_ids=None):
        self.entries = entries
        self.version = version
        self.language = language
        self.generator = generator
        self.prompt_version = prompt_version
        self.description_count = int(description_count)
        self.source_path = source_path
        self.metadata = metadata or {}
        # --- 四视角（Aff3DFunc 功能文本增强）---
        # views  : 视角名顺序，如 ['action','function','appearance','environment']
        # view_ids: dict[query_key] -> list[str]，与 entries[key] 逐条对应的视角名
        # v1 扁平池两者均为 None，行为与改动前一致。
        self.views = list(views) if views else None
        self.view_ids = view_ids or {}
        self._hash = self._compute_hash()

    # ------------------------------------------------------------------ 构造

    @classmethod
    def load(cls, path, description_count=DEFAULT_DESCRIPTION_COUNT,
             missing='error', required_keys=None, project_root=None):
        """从 JSON 加载描述池。

        Args:
            path: JSON 路径（相对路径按项目根解析）。
            description_count: 每个 query_key 期望的描述条数 M。
            missing: 'error' | 'report'，缺失/不足条目时的处理策略。
            required_keys: 需要覆盖的 query_key 列表（通常是数据集的功能类别表）。
            project_root: 项目根目录；None 时取本文件上两级。

        Returns:
            DescriptionBank
        """
        if path is None:
            raise DescriptionBankError("description_bank_path 为 None")

        if missing not in _VALID_MISSING_POLICIES:
            raise ValueError(f"missing 策略必须是 {_VALID_MISSING_POLICIES} 之一，收到 {missing!r}")

        resolved = cls._resolve_path(path, project_root)
        if not os.path.isfile(resolved):
            raise DescriptionBankError(f"描述池文件不存在: {resolved}")

        with open(resolved, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise DescriptionBankError(f"描述池文件顶层必须是 JSON object: {resolved}")
        if 'entries' not in raw:
            raise DescriptionBankError(f"描述池文件缺少 'entries' 字段: {resolved}")

        raw_entries = raw['entries']
        if isinstance(raw_entries, dict):
            items = raw_entries.items()
        elif isinstance(raw_entries, list):
            items = [(e.get('query_key'), e) for e in raw_entries]
        else:
            raise DescriptionBankError("'entries' 必须是 list 或 object")

        entries = {}
        view_ids = {}
        raw_views = raw.get('views')
        for key, entry in items:
            if key is None:
                raise DescriptionBankError(f"描述池条目缺少 query_key: {entry}")
            if key in entries:
                raise DescriptionBankError(f"描述池存在重复 query_key: {key}")

            if isinstance(entry, dict) and 'descriptions' not in entry:
                # v2 四视角结构：{view_name: [texts]}，展平并记录每条的视角
                order = [v for v in (raw_views or []) if v in entry]
                order += [v for v in entry.keys() if v not in order]
                flat, tags = [], []
                for v in order:
                    for text in cls._clean_descriptions(entry.get(v, [])):
                        flat.append(text)
                        tags.append(v)
                descriptions, this_view_ids = flat, tags
            else:
                # v1 扁平结构：{'query_key':..., 'descriptions': [...]} 或直接是 list
                raw_list = entry.get('descriptions', []) if isinstance(entry, dict) else entry
                descriptions = cls._clean_descriptions(raw_list)
                this_view_ids = None

            if not descriptions:
                raise DescriptionBankError(f"query_key '{key}' 没有任何非空描述")
            entries[key] = descriptions
            if this_view_ids:
                view_ids[key] = this_view_ids

        bank = cls(
            entries=entries,
            version=raw.get('version'),
            language=raw.get('language'),
            generator=raw.get('generator'),
            prompt_version=raw.get('prompt_version'),
            description_count=int(raw.get('description_count', description_count)),
            source_path=resolved,
            metadata={k: v for k, v in raw.items()
                      if k not in ('entries', 'version', 'language', 'generator',
                                   'prompt_version', 'description_count')},
            views=raw_views,
            view_ids=view_ids,
        )
        bank.check_integrity(required_keys=required_keys, missing=missing, verbose=True)
        return bank

    @staticmethod
    def _resolve_path(path, project_root):
        if os.path.isabs(path):
            return path
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, path)

    @staticmethod
    def _clean_descriptions(raw_list):
        """去空、去空白、去重（保序）。"""
        out = []
        seen = set()
        for item in raw_list:
            if not isinstance(item, str):
                continue
            text = ' '.join(item.split())
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    # ------------------------------------------------------------------ 查表

    def has(self, query_key):
        return query_key in self.entries

    def get(self, query_key):
        """返回该 query_key 的描述列表；不存在时抛 KeyError。

        调用方若希望「未知查询显式回退」，应先用 has() 判断并把
        description_valid 全置 False，而不是让此处静默返回空列表。
        """
        if query_key not in self.entries:
            raise KeyError(f"描述池缺少 query_key: {query_key}")
        return list(self.entries[query_key])

    def get_padded(self, query_key, description_count=None):
        """返回 (descriptions, valid) 两个长度固定的列表。

        - 条目存在：按 description_count 截断/以空串补齐，valid 标记真实条目。
        - 条目不存在：返回全空串 + 全 False，由调用方决定是报错还是零残差回退。
        """
        m = int(description_count or self.description_count)
        descriptions = [''] * m
        valid = [False] * m
        if query_key in self.entries:
            got = self.entries[query_key][:m]
            descriptions[:len(got)] = got
            for i in range(len(got)):
                valid[i] = True
        return descriptions, valid

    def get_view_ids(self, query_key, description_count=None):
        """与 get_padded 逐条对齐的视角名列表；v1 扁平池返回 None。"""
        m = int(description_count or self.description_count)
        tags = self.view_ids.get(query_key)
        if not tags:
            return None
        out = list(tags[:m])
        out += [''] * (m - len(out))
        return out

    def __len__(self):
        return len(self.entries)

    def __contains__(self, query_key):
        return query_key in self.entries

    # ------------------------------------------------------------------ 校验

    def check_integrity(self, required_keys=None, missing='error', verbose=False):
        """检查条数一致性与必需 key 覆盖。

        返回报告 dict；missing='error' 时发现问题直接抛错。
        """
        report = {
            'source_path': self.source_path,
            'version': self.version,
            'bank_hash': self.bank_hash,
            'num_entries': len(self.entries),
            'description_count': self.description_count,
            'short_entries': {},          # key -> 实际条数（不足 M）
            'long_entries': {},           # key -> 实际条数（超过 M，会被截断）
            'missing_keys': [],           # 必需但缺失
            'extra_keys': [],             # 银行里有但数据集用不到
            'views': self.views,          # 四视角名（v2）或 None
            'per_view_count': self.metadata.get('per_view_count'),
        }

        for key, descs in self.entries.items():
            n = len(descs)
            if n < self.description_count:
                report['short_entries'][key] = n
            elif n > self.description_count:
                report['long_entries'][key] = n

        if required_keys:
            required = set(required_keys)
            present = set(self.entries.keys())
            report['missing_keys'] = sorted(required - present)
            report['extra_keys'] = sorted(present - required)

        problems = []
        if report['short_entries']:
            problems.append(f"描述不足 M={self.description_count} 的条目: {report['short_entries']}")
        if report['missing_keys']:
            problems.append(f"必需 query_key 缺失: {report['missing_keys']}")

        if verbose:
            print(f"[DescriptionBank] 加载 {len(self.entries)} 个 query_key, M={self.description_count}, "
                  f"version={self.version}, hash={self.bank_hash[:12]}")
            print(f"[DescriptionBank] 文件: {self.source_path}")
            if self.views:
                print(f"[DescriptionBank] 四视角池: views={self.views}, "
                      f"每视角 {self.metadata.get('per_view_count')} 条, "
                      f"覆盖 {len(self.view_ids)}/{len(self.entries)} 个 key")
            if report['long_entries']:
                print(f"[DescriptionBank] 提示: 以下条目超过 M，将被截断: {report['long_entries']}")
            if report['extra_keys']:
                print(f"[DescriptionBank] 提示: 银行中存在数据集未使用的 key: {report['extra_keys']}")

        if problems:
            message = "；".join(problems)
            if missing == 'error':
                raise DescriptionBankError(f"描述池完整性检查失败（{self.source_path}）：{message}")
            print(f"[DescriptionBank] WARNING: {message}")

        return report

    # ------------------------------------------------------------------ 哈希

    def _compute_hash(self):
        """对规范化的条目内容取 sha256。

        只包含 version + 排序后的 (query_key, descriptions)，
        使「改动描述文本」必然导致哈希变化，而文件注释/顺序无关。
        """
        canonical = {
            'version': self.version,
            'description_count': self.description_count,
            'entries': {k: self.entries[k] for k in sorted(self.entries.keys())},
        }
        blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()

    @property
    def bank_hash(self):
        return self._hash

    @property
    def hash_short(self):
        return self._hash[:12]

    def summary(self):
        return {
            'path': self.source_path,
            'version': self.version,
            'language': self.language,
            'generator': self.generator,
            'prompt_version': self.prompt_version,
            'description_count': self.description_count,
            'num_entries': len(self.entries),
            'bank_hash': self.bank_hash,
        }
