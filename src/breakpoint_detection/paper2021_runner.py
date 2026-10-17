"""Evaluate frozen validation-selected baselines without search or training."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from .data import load_yahoo_csv
from .metrics import paper_event_metrics
from .protocol import SETTINGS, MARGINS, PACKAGE, empty_output, sha256, source_hashes
from .source_dataset import load_dataset

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def counts(tp, fp, fn):
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def aggregate(truths, predictions, margin):
    rows=[paper_event_metrics(t,p,margin) for t,p in zip(truths,predictions,strict=True)]
    return counts(*(sum(row[key] for row in rows) for key in ('tp','fp','fn')))


def local_peaks(scores):
    scores=np.asarray(scores,dtype=float); finite=np.isfinite(scores)
    values=np.where(finite,scores,-np.inf)
    mask=finite & (values>np.r_[-np.inf,values[:-1]]) & (values>=np.r_[values[1:],-np.inf])
    return np.where(mask,scores,np.nan)


def predict(peaks, threshold):
    return np.flatnonzero(np.isfinite(peaks) & (peaks>threshold))


def traditional_trajectory(job):
    model,path,candidate=job; config=SETTINGS[model]; window=candidate['window']
    series=load_yahoo_csv(path,'changepoint',config['preprocessing']); values=series.values
    if model=='rulsif':
        from .models.author_rulsif import score_series
        scores,_,_,available,plot=score_series(values,config['embedding'],n=window,alpha=config['alpha'])
        return dict(scores=scores,available=available,peaks=local_peaks(scores),
                    boundary_positions=np.flatnonzero(np.isfinite(scores)),plot_positions=plot)
    from .models.author_floss import AuthorStreamingFloss,pick_k_minimum
    number=int(series.name.rsplit('TS',1)[1])
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],number,window]))
    values=values+config['noise_std']*rng.standard_normal(len(values)); buffer=config['buffer']
    detector=AuthorStreamingFloss(values[:buffer],window,rng)
    scores=np.full(len(values),np.inf); available=np.full(len(values),-1,dtype=int)
    def record():
        scores[detector.center_report_position_]=detector.center_score_
        available[detector.center_report_position_]=detector.available_stop
    record()
    for value in values[buffer:]: detector.update(value); record()
    prediction=pick_k_minimum(scores,config['source_k'],config['suppression_factor']*window)
    return dict(scores=scores,available=available,prediction=prediction)


def execution_plan(model):
    """Unique fixed scoring jobs, each shared by its applicable margins."""
    plans = {}
    for margin in MARGINS:
        parameters = SETTINGS[model]['by_margin'][str(margin)]
        window = parameters['window']
        identity = f'w{window:03d}'
        if model == 'tscp2':
            identity += f"-b{parameters['batch_size']:03d}-c{parameters['code_size']:02d}-lr{parameters['learning_rate']:.0e}"
        job = {k: v for k, v in parameters.items() if k != 'threshold'}
        job['id'] = identity
        if identity in plans and plans[identity] != job:
            raise ValueError('conflicting fixed scoring settings')
        plans[identity] = job
    return list(plans.values())


def validate_settings():
    if SETTINGS['metric'] != 'event_micro_f1' or tuple(MARGINS) != (24, 50, 75):
        raise ValueError('2021 event evaluation and Yahoo margins must be retained')
    if SETTINGS['rulsif']['regularization'] != .01:
        raise ValueError('2021 fixes RuLSIF lambda=.01')
    if SETTINGS['floss']['distance'] != 'z-normalized Euclidean':
        raise ValueError('2021 requires FLOSS z-normalized Euclidean')
    for key, value in dict(stacks=2, filters=64, kernel_size=4, dilations=[1,4,16]).items():
        if SETTINGS['tscp2'][key] != value:
            raise ValueError(f'2021 fixes TSCP2 {key}={value}')
    for model in ('floss', 'rulsif', 'tscp2'):
        config = SETTINGS[model]
        if set(config['by_margin']) != {str(m) for m in MARGINS}:
            raise ValueError('each margin needs fixed parameters')
        if any(key in config for key in ('windows', 'batch_sizes', 'code_sizes', 'learning_rates', 'grid_dimensions')):
            raise ValueError('runtime parameter grids are not supported')
        for row in config['by_margin'].values():
            if row['window'] < 2:
                raise ValueError('invalid fixed window')
            if model != 'floss' and not np.isfinite(row['threshold']):
                raise ValueError('fixed finite threshold required')
            if model == 'rulsif' and (row['window'] < 5 or row['window'] % 5):
                raise ValueError('five-fold source port requires n to be a multiple of five')
        execution_plan(model)
    neural = SETTINGS['tscp2']
    checkpoint = PACKAGE.parents[1]/neural['checkpoint']
    if sha256(checkpoint) != neural['checkpoint_sha256']:
        raise ValueError('fixed checkpoint hash mismatch')
    return checkpoint


def trajectories(model, job, files, directory, workers, cache):
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    manifest = load_json(Path(cache)/'baseline-assets.json') if cache else None
    if manifest:
        if manifest['parameters'][model] != SETTINGS[model]:
            raise ValueError('cached generating parameters differ')
        key = f'models/author_{model}.py'
        if manifest['source_sha256'][key] != sha256(PACKAGE/key):
            raise ValueError('cached estimator source differs')
        for name, expected in manifest['preprocessing_sha256'].items():
            if sha256(PACKAGE/name) != expected:
                raise ValueError('cached preprocessing source differs')
        scorer = hashlib.sha256(inspect.getsource(traditional_trajectory).encode('utf-8')).hexdigest()
        if scorer != manifest['scorer_sha256']:
            raise ValueError('cached scoring implementation differs')
    for path in files:
        if manifest:
            relative = f'{model}/trajectories/{job["id"]}/{path.stem}.npz'
            source = Path(cache)/relative
            if manifest['input_sha256'].get(path.name) != sha256(path):
                raise ValueError('cached input hash mismatch')
            if sha256(source) != manifest['files'][relative]:
                raise ValueError('cached trajectory hash mismatch')
            with np.load(source, allow_pickle=False) as data:
                row = {key: data[key] for key in data.files}
            # Cached observations are reused; derive alarms afresh from scores.
            if model == 'rulsif':
                row['peaks'] = local_peaks(row['scores'])
            else:
                from .models.author_floss import pick_k_minimum
                config = SETTINGS['floss']
                row['prediction'] = pick_k_minimum(row['scores'], config['source_k'],
                    config['suppression_factor']*job['window'])
            row['reused_from'] = np.array(str(source.resolve()))
            row['reused_sha256'] = np.array(sha256(source))
            np.savez_compressed(directory/f'{path.stem}.npz', **row)
        else:
            pending.append((model, str(path), job))
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job_args, row in zip(pending, pool.map(traditional_trajectory, pending), strict=True):
                np.savez_compressed(directory/f'{Path(job_args[1]).stem}.npz', **row)
    rows = []
    for path in files:
        with np.load(directory/f'{path.stem}.npz', allow_pickle=False) as data:
            rows.append({key: data[key] for key in data.files})
    return rows


def neural_scores(files, series, directory, checkpoint, args):
    directory.mkdir(parents=True)
    input_path = directory/'input.npz'
    np.savez_compressed(input_path, values=np.stack([s.values for s in series]))
    subprocess.run([str(args.tscp2_python or sys.executable), str(PACKAGE/'paper2021_worker.py'),
        '--input', str(input_path), '--checkpoint', str(checkpoint), '--output', str(directory/'scores.npz'),
        '--device', args.tscp2_device], check=True,
        env=dict(os.environ, TSCP2_CPU_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='SEQUENTIAL'))
    with np.load(directory/'scores.npz', allow_pickle=False) as data:
        return [dict(peaks=p, scores=s, available=a) for p, s, a in
                zip(data['peaks'], data['scores'], data['available'], strict=True)]


def evaluate_phase(model, phase, files, series, output, checkpoint, args):
    truths = [np.flatnonzero(s.labels) for s in series]
    rows_by_window = {}
    for job in execution_plan(model):
        directory = output/phase/job['id']
        if model == 'tscp2':
            rows = neural_scores(files, series, directory, checkpoint, args)
        else:
            rows = trajectories(model, job, files, directory, args.workers, args.reuse_scores)
        rows_by_window[job['window']] = rows
    metrics, events, summary = [], [], []
    for margin in MARGINS:
        parameters = SETTINGS[model]['by_margin'][str(margin)]
        window = parameters['window']
        predictions = []
        for path, truth, row in zip(files, truths, rows_by_window[window], strict=True):
            # Never inspect labels to choose parameters or thresholds.
            prediction = row['prediction'] if model == 'floss' else predict(row['peaks'], parameters['threshold'])
            prediction = np.sort(prediction)
            predictions.append(prediction)
            metrics.append(dict(series=path.stem, model=model, phase=phase, detection_margin=margin,
                window=window, truth=json.dumps(truth.tolist()), prediction=json.dumps(prediction.tolist()),
                metric='event_f1', **paper_event_metrics(truth, prediction, margin)))
            for point in prediction:
                events.append(dict(series=path.stem, model=model, phase=phase, detection_margin=margin,
                    position=int(point), score=float(row['scores'][point]), score_available_stop=int(row['available'][point]),
                    selection_available_stop=len(row['scores']), evaluation_stop=len(row['scores']),
                    coordinate='literal_author_center' if model == 'floss' else 'history_future_boundary'))
        summary.append(dict(model=model, phase=phase, protocol=SETTINGS['version'], metric='event_micro_f1',
            detection_margin=margin, window=window, n_series=len(files), change_points=sum(map(len, truths)),
            **aggregate(truths, predictions, margin)))
    return metrics, events, summary


def run(args):
    checkpoint = validate_settings()
    output = Path(args.output).resolve()
    official = PACKAGE.parents[1]/'results/yahoo-a4'
    if output == official or official in output.parents:
        raise ValueError('output cannot overwrite official results')
    models = ['floss', 'rulsif', 'tscp2'] if args.model == 'all' else [args.model]
    files, series, _ = load_dataset(args.input, 'source-pooled')
    input_hashes = {str(p.resolve()): sha256(p) for p in files}
    source = source_hashes()
    if args.resume:
        meta = load_json(output/'run.json')
        if meta['source_sha256'] != source or meta['input_sha256'] != input_hashes or meta['models'] != models:
            raise ValueError('resume source, inputs or model set differs')
        if meta['status'] != 'complete':
            raise ValueError('incomplete fixed evaluation; choose a new empty output directory')
        for name, expected in meta['output_sha256'].items():
            if sha256(output/name) != expected:
                raise ValueError(f'completed output changed: {name}')
        return 0
    empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(status='scoring', models=models, settings=SETTINGS, source_sha256=source,
        input_sha256=input_hashes, checkpoint_sha256=sha256(checkpoint), started_at=datetime.now(timezone.utc).isoformat(),
        training_performed=False, parameter_selection_performed=False, new_blind_test=False)
    write_json(output/'run.json', meta)
    for model in models:
        destination = output/model
        destination.mkdir()
        write_json(destination/'fixed-settings.json', SETTINGS[model])
        metrics, events, summary = [], [], []
        for phase, region in [('validation', slice(40,60)), ('test', slice(60,100))]:
            m, e, s = evaluate_phase(model, phase, files[region], series[region], destination, checkpoint, args)
            metrics.extend(m); events.extend(e); summary.extend(s)
        pd.DataFrame(metrics).to_csv(destination/'metrics.csv', index=False)
        pd.DataFrame(events, columns=['series','model','phase','detection_margin','position','score','score_available_stop',
            'selection_available_stop','evaluation_stop','coordinate']).to_csv(destination/'events.csv', index=False)
        pd.DataFrame(summary).to_csv(destination/'summary.csv', index=False)
        print(pd.DataFrame(summary)[['model','phase','detection_margin','f1','tp','fp','fn']].to_string(index=False), flush=True)
    if source_hashes() != source or any(sha256(p) != h for p, h in input_hashes.items()) or sha256(checkpoint) != meta['checkpoint_sha256']:
        raise RuntimeError('sources, inputs or checkpoint changed')
    meta.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
        output_sha256={p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*') if p.is_file() and p.name != 'run.json'})
    write_json(output/'run.json', meta)
    return 0
