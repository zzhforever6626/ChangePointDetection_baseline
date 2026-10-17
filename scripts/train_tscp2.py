"""Train TSCP2 from scratch with the fixed settings, then verify the baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from breakpoint_detection.protocol import SETTINGS, empty_output, sha256
from breakpoint_detection.source_dataset import load_dataset
from breakpoint_detection.paper2021_runner import evaluate_phase
from breakpoint_detection.tscp2_adapter import training_pairs, separated_batches
from verify import counts


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(output):
    config = dict(SETTINGS['tscp2'])
    rows = config.pop('by_margin')
    parameters = {k: v for k, v in rows['24'].items() if k != 'threshold'}
    assert all({k: v for k, v in row.items() if k != 'threshold'} == parameters for row in rows.values())
    config.update(parameters)
    for key in ('checkpoint', 'checkpoint_sha256'):
        config.pop(key)
    files, series, _ = load_dataset(ROOT / 'data/A4Benchmark', preprocessing=config['preprocessing'])
    # Only TS1-40 contribute training pairs. TS41-60 are scored after training.
    np.savez_compressed(output / 'development-input.npz',
        values=np.stack([s.values for s in series[:60]]),
        train_stops=np.array([1680] * 40 + [0] * 20), validation_stops=np.full(60, 1680))
    write(output / 'training-config.json', config)
    return config


def verify(output):
    config = json.loads((output / 'training-config.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(output / 'trained/encoder.pt', map_location='cpu', weights_only=False)
    original = torch.load(ROOT / SETTINGS['tscp2']['checkpoint'], map_location='cpu', weights_only=False)
    assert checkpoint['parameters'] == config
    assert checkpoint['window'] == config['window']
    history = pd.read_csv(output / 'trained/training.csv')
    assert history.epoch.tolist() == list(range(1, checkpoint['epochs'] + 1))
    stalled = 0
    for i, row in enumerate(history.itertuples(index=False)):
        assert np.isfinite([row.loss, row.positive_similarity, row.negative_similarity]).all()
        stalled = stalled + 1 if i > 5 and (abs(row.loss - history.iloc[i-1].loss) < .0001 or history.iloc[i-1].loss < row.loss) else 0
        assert row.stalled == stalled
        assert stalled < 4 or i == len(history) - 1
    assert stalled == 4 or len(history) == config['max_epochs']
    _, series, _ = load_dataset(ROOT / 'data/A4Benchmark', preprocessing=config['preprocessing'])
    values = np.stack([s.values for s in series[:60]])
    stops = np.array([1680] * 40 + [0] * 20)
    with np.load(output / 'development-input.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['values'], values, strict=True)
        np.testing.assert_array_equal(data['train_stops'], stops, strict=True)
    pairs, locations = training_pairs(values, stops, config['window'], config['training_stride'], last_start_inclusive=False)
    batches = separated_batches(locations, config['batch_size'], 2 * config['window'], config['seed'])
    with np.load(output / 'trained/training_samples.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['locations'], locations, strict=True)
        np.testing.assert_array_equal(data['batches'], batches, strict=True)
    assert locations[:, 0].max() < 40 and checkpoint['batch_count'] == len(batches)
    assert checkpoint['state_dict'].keys() == original['state_dict'].keys()
    weights_equal = all(a.dtype == original['state_dict'][k].dtype and a.shape == original['state_dict'][k].shape
        and a.numpy().tobytes() == original['state_dict'][k].numpy().tobytes() for k, a in checkpoint['state_dict'].items())
    history_columns = ['epoch', 'loss', 'positive_similarity', 'negative_similarity', 'stalled']
    reference_history = pd.read_csv(ROOT / 'logs/tscp2-training/training.csv')
    history_equal = history[history_columns].equals(reference_history[history_columns])
    destination = output / 'evaluation/tscp2'
    empty_output(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files, series, _ = load_dataset(ROOT / 'data/A4Benchmark', preprocessing=config['preprocessing'])
    metrics, events, summaries = [], [], []
    args = SimpleNamespace(tscp2_python=Path(sys.executable), tscp2_device='cpu')
    for phase, region in [('validation', slice(40, 60)), ('test', slice(60, 100))]:
        m, e, s = evaluate_phase('tscp2', phase, files[region], series[region], destination, output / 'trained/encoder.pt', args)
        metrics.extend(m); events.extend(e); summaries.extend(s)
    metrics, events, summaries = map(pd.DataFrame, (metrics, events, summaries))
    for name, frame in [('metrics', metrics), ('events', events), ('summary', summaries)]:
        frame.to_csv(destination / (name + '.csv'), index=False)
    original_metrics = pd.read_csv(ROOT / 'results/yahoo-a4/tscp2/metrics.csv')
    metrics_equal = all(metrics[k].tolist() == original_metrics[k].tolist()
        for k in ('series', 'phase', 'detection_margin', 'truth', 'prediction', 'tp', 'fp', 'fn'))
    for row in metrics.itertuples(index=False):
        truth, prediction = json.loads(row.truth), json.loads(row.prediction)
        raw = pd.read_csv(ROOT / 'data/A4Benchmark' / (row.series + '.csv'))
        assert truth == np.flatnonzero(raw.changepoint.to_numpy()).tolist()
        assert counts(truth, prediction, row.detection_margin) == (row.tp, row.fp, row.fn)
    for row in summaries.itertuples(index=False):
        group = metrics[(metrics.phase == row.phase) & (metrics.detection_margin == row.detection_margin)]
        tp, fp, fn = (int(group[k].sum()) for k in ('tp', 'fp', 'fn'))
        assert (tp, fp, fn) == (row.tp, row.fp, row.fn)
        assert abs(row.f1 - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    report = dict(status='passed' if weights_equal and history_equal and metrics_equal else 'baseline_mismatch',
        checked_at=datetime.now(timezone.utc).isoformat(), seed=config['seed'], epochs=len(history),
        training_pairs=len(pairs), batch_count=len(batches), weight_tensors=len(checkpoint['state_dict']),
        all_weight_tensors_equal=weights_equal, training_history_equal=history_equal,
        predictions_and_counts_equal=metrics_equal, independent_metric_rows=len(metrics), summary_rows=len(summaries),
        test_f1=summaries[summaries.phase == 'test'].f1.tolist(),
        parameter_selection_performed=False, checkpoint_sha256=sha256(output / 'trained/encoder.pt'))
    write(output / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true', help='verify an existing completed training output')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify_only:
        return verify(output)
    empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    config = prepare(output)
    command = [sys.executable, str(ROOT / 'src/breakpoint_detection/models/tscp2.py'),
        '--input', str(output / 'development-input.npz'), '--config', str(output / 'training-config.json'),
        '--output', str(output / 'trained'), '--window', str(config['window']), '--device', 'cpu']
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
        TSCP2_CPU_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_THREADING_LAYER='SEQUENTIAL')
    process = dict(command=command, cwd=str(ROOT), started_at=datetime.now(timezone.utc).isoformat(),
        initialized_from_checkpoint=False, seed=config['seed'], device='cpu', threads=2)
    write(output / 'process.json', process)
    with (output / 'stdout.log').open('w', encoding='utf-8') as out, (output / 'stderr.log').open('w', encoding='utf-8') as err:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err)
    process.update(exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    write(output / 'process.json', process)
    if result.returncode:
        raise RuntimeError(f'training failed; see {output / "stderr.log"}')
    return verify(output)


if __name__ == '__main__':
    raise SystemExit(main())
