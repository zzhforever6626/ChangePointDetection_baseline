"""Run the fixed baselines and preserve actual process logs and every test row."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from breakpoint_detection.protocol import sha256

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def runtime():
    code = '''import torch
import sys,platform,json,numpy,pandas
print(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
    numpy=numpy.__version__,pandas=pandas.__version__,torch=torch.__version__,device='cpu')))
'''
    return json.loads(subprocess.check_output([sys.executable,'-c',code], text=True, encoding='utf-8'))

def compact_run(directory):
    """Keep the complete output tables while discarding intermediate arrays."""
    directory = Path(directory).resolve()
    meta_path = directory/'run.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    arrays = list(directory.rglob('*.npz'))
    for path in arrays:
        if not path.resolve().is_relative_to(directory):
            raise ValueError('intermediate array outside run directory')
        path.unlink()
    for path in sorted((p for p in directory.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    meta['output_sha256'] = {name:value for name,value in meta['output_sha256'].items()
                             if not name.endswith('.npz')}
    meta['output_layout'] = 'metrics-only'
    write(meta_path,meta)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--fresh-rulsif',action='store_true',help='recompute all density scores instead of verified cached scores')
    parser.add_argument('--retrain-tscp2',action='store_true',help='train TSCP2 from scratch and verify its new checkpoint before evaluating all models')
    parser.add_argument('--keep-scores',action='store_true',help='keep intermediate score arrays in the run output')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('choose a new or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='SEQUENTIAL')
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),
        environment=runtime(),
        fixed_settings_sha256=sha256(ROOT/'src/breakpoint_detection/baseline_settings.json'),
        scope='fixed baseline evaluation',
        modes=dict(floss='fresh full validation/test scoring',rulsif='fresh full scoring' if args.fresh_rulsif else 'hash-verified score replay; alarms and metrics recomputed',
            tscp2='fixed checkpoint reload and inference; no training'))
    write(output/'environment.json',metadata)
    if args.retrain_tscp2:
        command = [sys.executable, str(ROOT/'scripts/train_tscp2.py'), '--output', str(output/'tscp2-training')]
        with (output/'tscp2-training.stdout.log').open('w',encoding='utf-8') as stdout, \
             (output/'tscp2-training.stderr.log').open('w',encoding='utf-8') as stderr:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,check=True)
        metadata['training_verification'] = json.loads((output/'tscp2-training/verification.json').read_text(encoding='utf-8'))
        write(output/'environment.json',metadata)
        print('tscp2: from-scratch training and new-checkpoint verification passed',flush=True)
    for model in ('floss','rulsif','tscp2'):
        destination = output/model
        destination.mkdir()
        command = [sys.executable,'-m','breakpoint_detection.cli','data/A4Benchmark','--model',model,
            '--workers',str(args.workers),'--tscp2-device','cpu','--output',str(destination/'run')]
        if model=='rulsif' and not args.fresh_rulsif:
            command += ['--reuse-scores','results/yahoo-a4']
        process_meta = dict(command=command,cwd=str(ROOT),started_at=datetime.now(timezone.utc).isoformat(),
            mode=metadata['modes'][model])
        write(destination/'process.json',process_meta)
        # Capture each process's output streams.
        with (destination/'stdout.log').open('w',encoding='utf-8',newline='\n') as stdout, \
             (destination/'stderr.log').open('w',encoding='utf-8',newline='\n') as stderr:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        process_meta.update(exit_code=result.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        write(destination/'process.json',process_meta)
        if result.returncode:
            raise RuntimeError(f'{model} failed; see {destination}/stderr.log')
        # Export complete per-file test predictions and metrics.
        import pandas as pd
        metrics = pd.read_csv(destination/'run'/model/'metrics.csv')
        with (destination/'test-output.jsonl').open('w',encoding='utf-8',newline='\n') as log:
            for row in metrics[metrics.phase=='test'].to_dict(orient='records'):
                for key in ('truth','prediction'):
                    row[key] = json.loads(row[key])
                if pd.isna(row['mean_delay']):
                    row['mean_delay'] = None
                log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if not args.keep_scores:
            compact_run(destination/'run')
        print(f'{model}: exit=0, complete test rows={sum(metrics.phase=="test")}',flush=True)
    metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),status='complete')
    write(output/'environment.json',metadata)
    from verify import verify
    report = verify(output)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
