"""Audit all test outputs independently and compare with the fixed baseline."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from breakpoint_detection.protocol import sha256, SETTINGS, REFERENCE_F1

def counts(truth, prediction, margin):
    truth,prediction = np.sort(truth),np.sort(prediction)
    distance = np.abs(prediction[:,None]-truth[None,:])
    tp = 0
    while distance.size and distance.min()<margin:
        i,j = np.unravel_index(distance.argmin(),distance.shape)
        distance[i,:] = margin
        distance[:,j] = margin
        tp += 1
    return tp,len(prediction)-tp,len(truth)-tp

def verify(output):
    output = Path(output).resolve()
    checked = 0
    comparisons = []
    for model in ('floss','rulsif','tscp2'):
        folder = output/model
        process = json.loads((folder/'process.json').read_text(encoding='utf-8'))
        assert process['exit_code']==0,model
        assert (folder/'stdout.log').exists() and (folder/'stderr.log').exists()
        run = folder/'run'
        meta = json.loads((run/'run.json').read_text(encoding='utf-8'))
        assert meta['status']=='complete'
        assert meta['settings']==SETTINGS
        for name,expected in meta['output_sha256'].items():
            assert sha256(run/name)==expected,name
        for name,expected in meta['source_sha256'].items():
            assert sha256(ROOT/'src/breakpoint_detection'/name)==expected,name
        metrics = pd.read_csv(run/model/'metrics.csv')
        summary = pd.read_csv(run/model/'summary.csv')
        expected = pd.read_csv(ROOT/'results/yahoo-a4'/model/'summary.csv')
        logs = [json.loads(line) for line in (folder/'test-output.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(logs)==120
        assert len(metrics)==180
        for row in metrics.itertuples(index=False):
            truth,prediction = json.loads(row.truth),json.loads(row.prediction)
            raw = pd.read_csv(ROOT/'data/A4Benchmark'/f'{row.series}.csv')
            assert truth==np.flatnonzero(raw.changepoint.to_numpy()).tolist()
            assert counts(truth,prediction,row.detection_margin)==(row.tp,row.fp,row.fn)
            checked += 1
        tests = metrics[metrics.phase=='test'].to_dict(orient='records')
        for log,row in zip(logs,tests,strict=True):
            for key in ('series','detection_margin','window','tp','fp','fn'):
                assert log[key]==row[key]
            for key in ('truth','prediction'):
                assert log[key]==json.loads(row[key])
        old_metrics = pd.read_csv(ROOT/'results/yahoo-a4'/model/'metrics.csv')
        for key in ('series','phase','detection_margin','window','truth','prediction','tp','fp','fn'):
            assert metrics[key].tolist()==old_metrics[key].tolist(),(model,key)
        for row in summary.itertuples(index=False):
            group = metrics[(metrics.phase==row.phase)&(metrics.detection_margin==row.detection_margin)]
            tp,fp,fn = (int(group[key].sum()) for key in ('tp','fp','fn'))
            assert (tp,fp,fn)==(row.tp,row.fp,row.fn)
            assert tp+fn==(43 if row.phase=='validation' else 79)
            assert abs(row.f1-2*tp/(2*tp+fp+fn))<1e-12
            baseline = expected[(expected.phase==row.phase)&(expected.detection_margin==row.detection_margin)].iloc[0]
            assert abs(row.f1-baseline.f1)<1e-12
            if row.phase=='test':
                comparisons.append(dict(model=model,margin=row.detection_margin,baseline_f1=baseline.f1,
                    reproduced_f1=row.f1,absolute_difference=abs(row.f1-baseline.f1),tp=tp,fp=fp,fn=fn,
                    registered_paper_reference_f1=REFERENCE_F1[model][row.detection_margin],
                    difference_from_registered_paper_f1=row.f1-REFERENCE_F1[model][row.detection_margin],
                    exact_predictions=True))
    pd.DataFrame(comparisons).to_csv(output/'metric-comparison.csv',index=False)
    lines = ['# 复现指标对照','', '| 模型 | 容差 | 固定baseline F1 | 本次复现 F1 | 绝对差 | 项目登记论文参考F1 | TP / FP / FN |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['model']} | {row['margin']} | {row['baseline_f1']:.10f} | {row['reproduced_f1']:.10f} | {row['absolute_difference']:.10f} | {row['registered_paper_reference_f1']:.4f} | {row['tp']} / {row['fp']} / {row['fn']} |")
    lines += ['', '固定baseline的所有验证／测试逐文件预测也一致。论文参考值沿用protocol.py中REFERENCE_F1的项目登记，仅作对照，未用于选择配置；此表不表示复现了论文数值。', '']
    (output/'metric-comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    report = dict(status='passed',independent_metric_rows=checked,complete_test_log_rows=360,
        summary_rows=18,test_comparisons=9,all_test_predictions_identical=True,
        max_absolute_f1_difference=max(r['absolute_difference'] for r in comparisons))
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
