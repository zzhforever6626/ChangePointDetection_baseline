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

def runtime(interpreter, neural=False):
    code = '''import sys,platform,json,importlib.metadata as m
import numpy
v=dict(python=sys.version,platform=platform.platform(),numpy_runtime=numpy.__version__,numpy_distribution_metadata=m.version('numpy'))
'''
    if neural:
        code = 'import torch\n'+code+"v.update(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())\n"
    else:
        code += "import pandas\nv.update(pandas=pandas.__version__)\n"
    code += 'print(json.dumps(v))'
    return json.loads(subprocess.check_output([str(interpreter),'-c',code], text=True, encoding='utf-8'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tscp2-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--fresh-rulsif',action='store_true',help='recompute all density scores instead of verified cached scores')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('choose a new or empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='SEQUENTIAL')
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),
        baseline_environment=runtime(sys.executable),neural_environment=runtime(args.tscp2_python,True),
        fixed_settings_sha256=sha256(ROOT/'src/breakpoint_detection/baseline_settings.json'),
        scope='new actual execution in exported repository; no historical console output reconstructed',
        modes=dict(floss='fresh full validation/test scoring',rulsif='fresh full scoring' if args.fresh_rulsif else 'hash-verified score replay; alarms and metrics recomputed',
            tscp2='fixed checkpoint reload and inference; no training'))
    write(output/'environment.json',metadata)
    for model in ('floss','rulsif','tscp2'):
        destination = output/model
        destination.mkdir()
        command = [sys.executable,'-m','breakpoint_detection.cli','data/A4Benchmark','--model',model,
            '--workers',str(args.workers),'--tscp2-python',str(args.tscp2_python.resolve()),
            '--tscp2-device','cpu','--output',str(destination/'run')]
        if model=='rulsif' and not args.fresh_rulsif:
            command += ['--reuse-scores','results/yahoo-a4']
        process_meta = dict(command=command,cwd=str(ROOT),started_at=datetime.now(timezone.utc).isoformat(),
            mode=metadata['modes'][model])
        write(destination/'process.json',process_meta)
        # Never synthesize or overwrite process stdout/stderr.
        with (destination/'stdout.log').open('w',encoding='utf-8',newline='\n') as stdout, \
             (destination/'stderr.log').open('w',encoding='utf-8',newline='\n') as stderr:
            result = subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr)
        process_meta.update(exit_code=result.returncode,finished_at=datetime.now(timezone.utc).isoformat())
        write(destination/'process.json',process_meta)
        if result.returncode:
            raise RuntimeError(f'{model} failed; see {destination}/stderr.log')
        # Full per-file output is emitted from the just-completed run, not from baseline results.
        import pandas as pd
        metrics = pd.read_csv(destination/'run'/model/'metrics.csv')
        with (destination/'test-output.jsonl').open('w',encoding='utf-8',newline='\n') as log:
            for row in metrics[metrics.phase=='test'].to_dict(orient='records'):
                for key in ('truth','prediction'):
                    row[key] = json.loads(row[key])
                if pd.isna(row['mean_delay']):
                    row['mean_delay'] = None
                log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(f'{model}: exit=0, complete test rows={sum(metrics.phase=="test")}',flush=True)
    metadata.update(finished_at=datetime.now(timezone.utc).isoformat(),status='complete')
    write(output/'environment.json',metadata)
    from verify import verify
    report = verify(output)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
