"""Compare completed, matched-subset Stage A diagnostic experiments."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    experiments = {'learned_original': root.parent / 'diagnosis_2000',
                   **{name: root / name for name in ('fixed007', 'fixed020', 'learnable020', 'fresh_k1', 'fresh_k8')}}
    rows, traces = [], {}
    expected_subset = None
    expected_arrays = None
    expected_training = None
    for name, folder in experiments.items():
        report = json.loads((folder / 'report.json').read_text())
        subset = json.loads((folder / 'subset.json').read_text())
        if expected_subset is None:
            expected_subset = subset
        if subset != expected_subset:
            raise ValueError(f'{name}: subset/cue mismatch invalidates comparison')
        training = (report['source_checkpoint'], report['learning_rate'])
        with np.load(folder / 'predictions.npz') as saved:
            arrays = {key: saved[key] for key in ('points', 'masks', 'valid')}
        if expected_arrays is None:
            expected_arrays, expected_training = arrays, training
        if training != expected_training or any(not np.array_equal(arrays[key], expected_arrays[key]) for key in arrays):
            raise ValueError(f'{name}: unequal data or training configuration')
        if report['steps'] != 2000 or report['objects'] != 16 or report['all_gradient_steps_finite'] != 2000:
            raise ValueError(f'{name}: incomplete or unequal experiment budget')
        traces[name] = json.loads((folder / 'trace.json').read_text())
        final = report['final']
        counts = final['selected_basis_counts']
        rows.append({'experiment': name, 'loss': final['loss'],
                     'segmentation_loss': final['loss_components']['segmentation'],
                     'gt_mae': report['fit_reference']['prediction_gt_mae'],
                     'prediction_pair_mae': final['prediction_pair_mae'],
                     'collapsed_objects': sum(r['prediction_pair_mae'] < 1e-6 for r in final['per_object']),
                     'dominant_basis_fraction': max(counts)/sum(counts),
                     'basis_counts': counts, 'entropy': final['alpha_entropy_normalized'],
                     'temperature': final['selector_temperature_effective'],
                     'shuffled_loss_increase': final['permute']['loss']-final['loss'],
                     'last_100_selector_grad_median': float(np.median([r['gradient_norms_before_clip']['functional_basis_head.basis_selector'] for r in traces[name][-100:]]))})
    summary = {'scope': 'single-seed fixed training subset; not generalization or LAS comparison',
               'matched_subsets_verified': True, 'rows': rows}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, trace in traces.items():
        values = [np.mean([x['loss'] for x in trace[i:i+4]]) for i in range(0, len(trace), 4)]
        axes[int(name.startswith('fresh'))].plot(range(4, len(trace)+1, 4), values, label=name, linewidth=1)
    for ax, title in zip(axes, ('Temperature interventions: same trained head', 'Basis count: fresh heads, same trunk')):
        ax.set(xlabel='Optimizer step', ylabel='Four-batch mean loss', title=title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root / 'comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='work/stage_a/ablations')
    summarize(Path(parser.parse_args().root))
