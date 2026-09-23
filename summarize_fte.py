"""汇总 V1-A.FTE 三档消融进度：每 epoch 的 Best Val aIoU / 选择熵 / 在线方差。

用法：
    python summarize_fte.py                 # 默认读 runs_fte/
    python summarize_fte.py --dir /path
    python summarize_fte.py --last 10
"""

import argparse
import os
import re
import sys

TAGS = [
    ('b1', 'mean+pooled'),
    ('b2', 'cond+pooled'),
    ('b3', 'mean+concat'),
]

RE_AIOU = re.compile(r'Best Val aIoU:\s*([0-9.]+)')
RE_ENT = re.compile(
    r'\[V1-A\] 描述选择: 归一化熵=([0-9.]+).*?max_alpha=([0-9.]+)')
RE_ONV = re.compile(r'加权后描述嵌入区域间方差=([0-9.]+)')
RE_PROG = re.compile(r'Epoch (\d+):\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)')


def read_lines(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    return raw.replace('\r', '\n').split('\n')


def parse(path):
    lines = read_lines(path)
    if lines is None:
        return None
    aiou, ent, maxa, onv = [], [], [], []
    cur_epoch, cur_pct = None, 0
    for ln in lines:
        m = RE_AIOU.search(ln)
        if m:
            aiou.append(float(m.group(1)))
        m = RE_ENT.search(ln)
        if m:
            ent.append(float(m.group(1)))
            maxa.append(float(m.group(2)))
        m = RE_ONV.search(ln)
        if m:
            onv.append(float(m.group(1)))
        m = RE_PROG.search(ln)
        if m:
            cur_epoch = int(m.group(1))
            cur_pct = int(m.group(2))
    return {'aiou': aiou, 'ent': ent, 'maxa': maxa, 'onv': onv,
            'epoch': cur_epoch, 'pct': cur_pct}


def fmt(v, w=7):
    return (' ' * w) if v is None else ('%.*f' % (4, v)).rjust(w)


def verdict_of(e):
    if e is None:
        return 'n/a'
    return '均匀(未选择)' if e > 0.97 else ('塌缩' if e < 0.35 else '有区分度')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='runs_fte')
    ap.add_argument('--last', type=int, default=0)
    args = ap.parse_args()

    data = {t: parse(os.path.join(args.dir, t + '.log')) for t, _ in TAGS}

    print('=' * 96)
    print('V1-A.FTE 三档消融进度  (日志目录: %s)' % args.dir)
    print('=' * 96)
    for tag, mode in TAGS:
        d = data[tag]
        if d is None:
            print('%-4s %-14s 日志不存在' % (tag, mode))
            continue
        best = max(d['aiou']) if d['aiou'] else 0.0
        le = d['ent'][-1] if d['ent'] else None
        lo = d['onv'][-1] if d['onv'] else None
        print('%-4s %-14s %3d epoch | Best aIoU=%.4f | 熵=%s (%s) | 在线方差=%s | Epoch %s %d%%'
              % (tag, mode, len(d['aiou']), best,
                 '%.4f' % le if le is not None else ' n/a ', verdict_of(le),
                 '%.4f' % lo if lo is not None else ' n/a ',
                 d['epoch'] if d['epoch'] is not None else '-', d['pct']))

    print('-' * 96)
    n_epoch = max((len(data[t]['aiou']) for t, _ in TAGS if data[t]), default=0)
    if n_epoch == 0:
        print('还没有任何 epoch 完成')
        return 0

    start = max(0, n_epoch - args.last) if args.last else 0
    header = ' | '.join('%s aIoU 熵 在线V' % t.upper() for t, _ in TAGS)
    print('%6s | %s' % ('epoch', header))
    print('-' * 96)
    for i in range(start, n_epoch):
        cells = []
        for tag, _ in TAGS:
            d = data[tag]
            a = d['aiou'][i] if i < len(d['aiou']) else None
            e = d['ent'][i] if i < len(d['ent']) else None
            v = d['onv'][i] if i < len(d['onv']) else None
            cells.append('%s %s %s' % (fmt(a), fmt(e), fmt(v)))
        print('%6d | %s' % (i, ' | '.join(cells)))

    print('-' * 96)
    print('熵读法:   1.0=完全均匀(未选择)  0=塌缩到单条  0.35~0.97=有区分度')
    print('在线方差: 加权后描述嵌入在区域间的离散度；mean/concat 理论为 0，')
    print('          conditional 显著大于 0 说明条件加权保住了多样性')
    print('          （即 Aff3DFunc 批评的「池化方差坍缩」在本方法下不成立）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
