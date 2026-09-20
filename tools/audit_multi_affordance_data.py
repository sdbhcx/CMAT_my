"""Usage: python -m tools.audit_multi_affordance_data --help"""
import argparse
import json
from pathlib import Path

from data.multi_affordance_audit import audit_manifest
from data.object_affordance_index import read_json


def parser():
    result = argparse.ArgumentParser(description='Audit an explicit object/image/mask manifest, without guessing instance IDs.')
    result.add_argument('--manifest', help='Verified mapping manifest; omit for raw file-list inventory')
    result.add_argument('--dataset', choices=['piad', 'piadv2'])
    result.add_argument('--max-files', type=int, help='Raw mode only: sample at most this many files PER partition')
    result.add_argument('--workers', type=int, default=4, help='Raw point-file IO threads (bounded prefetch)')
    result.add_argument('--data-root', required=True)
    result.add_argument('--split')
    result.add_argument('--setting')
    result.add_argument('--output-dir')
    result.add_argument('--max-distance', type=float, default=None,
                        help='Explicit bidirectional alignment tolerance in raw coordinate units; default exact only')
    result.add_argument('--canonical-count', type=int, default=2048)
    result.add_argument('--min-multi-objects', type=int, default=1,
                        help='Project-specific Go/No-Go threshold; 1 is only a technical smoke-test threshold')
    return result


def run(args, index_path=None):
    if not args.manifest:
        if index_path is not None or not args.dataset:
            raise ValueError('Building an index requires --manifest; raw audit requires --dataset')
        from data.raw_affordance_audit import audit_raw_dataset
        output = args.output_dir or Path('work/data_audit') / args.dataset / 'all'
        summary = audit_raw_dataset(args.data_root, args.dataset, output, args.max_files, args.workers)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    if not args.split or not args.setting or args.max_files:
        raise ValueError('Manifest mode requires --split/--setting and does not support --max-files')
    manifest = read_json(args.manifest)
    # Restrict directory labels even when the report directory is explicitly set.
    for value in (manifest['dataset'], args.setting, args.split):
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
            raise ValueError('dataset/setting/split must be simple directory labels')
    output = args.output_dir or Path('work/data_audit') / manifest['dataset'] / args.setting / args.split
    summary = audit_manifest(manifest, args.data_root, args.split, args.setting, output,
                             args.max_distance, args.canonical_count, args.min_multi_objects, index_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    run(parser().parse_args())
