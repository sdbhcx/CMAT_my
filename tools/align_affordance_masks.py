"""Inspect/replay alignment independently of dataset indexing."""
import argparse
import json
from pathlib import Path
import numpy as np
from data.affordance_alignment import align_masks
from data.object_affordance_index import read_point_source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align source masks to canonical coordinates; writes npz and quality JSON')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--canonical', required=True, help='Root-relative point source')
    parser.add_argument('--source', required=True)
    parser.add_argument('--format', choices=['piad_txt', 'piadv2_npy'], required=True)
    parser.add_argument('--max-distance', type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    canonical, _ = read_point_source(args.data_root, {'path': args.canonical, 'format': args.format})
    points, masks = read_point_source(args.data_root, {'path': args.source, 'format': args.format})
    aligned, ids, weights, quality = align_masks(canonical, points, masks, args.max_distance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=canonical, masks=aligned, indices=ids, weights=weights)
    output.with_suffix('.json').write_text(json.dumps(quality, indent=2) + '\n')
    print(json.dumps(quality, indent=2))
