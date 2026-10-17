"""Fixed inference, label independence, artifact guards and event evaluation."""
import json
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from breakpoint_detection.metrics import paper_event_metrics
from breakpoint_detection.paper2021_runner import aggregate, execution_plan, predict, validate_settings
from breakpoint_detection.protocol import SETTINGS
from breakpoint_detection.tscp2_adapter import training_pairs, separated_batches, similarity_drop, phase_peak_scores

def test_event_duplicates_strict_boundary_and_closest_match():
    measured=paper_event_metrics([100],[99,101,101,124],24)
    assert (measured['tp'],measured['fp'],measured['fn'],measured['mean_delay'])==(1,3,0,1.)
    assert paper_event_metrics([100],[124],24)['tp']==0
    assert paper_event_metrics([100,110],[100],24)['fn']==1
    assert paper_event_metrics([100],[100,100],24)['f1']==pytest.approx(2/3)
    # Distinguishes registered closest-first overlap completion from max-cardinality DP.
    assert paper_event_metrics([0,4],[3,7],4)['tp']==1


def test_paper_sampling_and_history_are_preserved():
    values=np.arange(3*100).reshape(3,100)
    pairs,locations=training_pairs(values,[80,0,0],10,5,last_start_inclusive=False)
    assert all(i==0 and start+20<=80 for i,start in locations)
    batches=separated_batches(np.array([(i,j) for i in range(5) for j in range(0,200,5)]),16,40,0)
    for batch in batches:
        sites=np.array([(i,j) for i in range(5) for j in range(0,200,5)])[batch]
        for i,left in enumerate(sites):
            assert all(left[0]!=r[0] or abs(left[1]-r[1])>=40 for r in sites[i+1:])
    similarity=np.array([1.,1.,1.,0.,.5,1.])
    score=similarity_drop(similarity,3)
    assert np.isnan(score[:3]).all() and score[3]==1.
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(phase_peak_scores([0.,1.,2.,1.,0.],1,5))),[2])


def test_fixed_parameters_and_strict_thresholds():
    validate_settings()
    assert [r['window'] for r in execution_plan('floss')] == [25,55]
    assert [r['window'] for r in execution_plan('rulsif')] == [25,10]
    assert execution_plan('tscp2') == [dict(id='w032-b032-c16-lr3e-04', window=32,
        batch_size=32, code_size=16, learning_rate=.0003)]
    assert predict(np.array([np.nan,.5,.6]),.5).tolist() == [2]


def test_run_uses_fixed_thresholds_regardless_of_labels_and_guards_resume(tmp_path,monkeypatch):
    import breakpoint_detection.paper2021_runner as runner
    from breakpoint_detection.cli import parser
    files,series=[],[]
    for i in range(100):
        path=tmp_path/f'A4Benchmark-TS{i+1}.csv'
        path.write_text('fixture',encoding='utf-8');files.append(path)
        labels=np.zeros(100,int);labels[40]=1
        series.append(SimpleNamespace(name=path.stem,values=np.arange(100.),labels=labels))
    monkeypatch.setattr(runner,'load_dataset',lambda *a:(files,series,[]))
    def scores(n):
        peaks=np.full(100,np.nan);peaks[[20,40,70]]=[.1,60.,.6]
        return [dict(scores=peaks.copy(),available=np.arange(100)+10,peaks=peaks.copy(),
            prediction=np.array([20,40,70])) for _ in range(n)]
    calls=[]
    def fake_trajectories(model,job,selected_files,directory,*args):
        calls.append((model,job['window'],len(selected_files)))
        return scores(len(selected_files))
    def fake_neural(selected_files,selected_series,directory,*args):
        return scores(len(selected_files))
    monkeypatch.setattr(runner,'trajectories',fake_trajectories)
    monkeypatch.setattr(runner,'neural_scores',fake_neural)
    previous=None
    for trial in range(2):
        if trial:
            for s in series[40:]:s.labels[:]=0;s.labels[[0,99]]=1
        output=tmp_path/f'run{trial}'
        args=parser().parse_args(['unused','--output',str(output),'--workers','1'])
        assert runner.run(args)==0
        predictions={}
        for model in ('floss','rulsif','tscp2'):
            metrics=pd.read_csv(output/model/'metrics.csv')
            predictions[model]=metrics.prediction.tolist()
            assert len(metrics)==180
            assert json.loads((output/model/'fixed-settings.json').read_text())==SETTINGS[model]
            summary=pd.read_csv(output/model/'summary.csv')
            assert summary.metric.eq('event_micro_f1').all()
            assert summary[summary.phase=='test'].change_points.eq(80 if trial else 40).all()
        if previous is not None:assert predictions==previous
        previous=predictions
        assert len(calls)==8*(trial+1)  # two windows per traditional model, two phases
        meta=json.loads((output/'run.json').read_text())
        assert not meta['training_performed'] and not meta['parameter_selection_performed']
        args.resume=True
        assert runner.run(args)==0
        with (output/'floss/summary.csv').open('a') as f:f.write('tampered')
        with pytest.raises(ValueError,match='completed output changed'):runner.run(args)


def test_cached_trajectory_rejects_input_and_payload_tampering(tmp_path):
    import breakpoint_detection.paper2021_runner as runner
    from breakpoint_detection.protocol import PACKAGE, sha256
    source=tmp_path/'A4Benchmark-TS41.csv';source.write_text('fixture')
    relative='rulsif/trajectories/w025/A4Benchmark-TS41.npz'
    cache=tmp_path/'cache';payload=cache/relative;payload.parent.mkdir(parents=True)
    np.savez_compressed(payload,scores=np.array([1.,2.]))
    key='models/author_rulsif.py'
    runner.write_json(cache/'baseline-assets.json',dict(parameters={'rulsif':SETTINGS['rulsif']},
        input_sha256={source.name:sha256(source)},source_sha256={key:sha256(PACKAGE/key)},
        preprocessing_sha256={'data.py':sha256(PACKAGE/'data.py')},
        scorer_sha256=hashlib.sha256(inspect.getsource(runner.traditional_trajectory).encode('utf-8')).hexdigest(),
        files={relative:sha256(payload)}))
    args=('rulsif',execution_plan('rulsif')[0],[source],tmp_path/'out',1,cache)
    assert len(runner.trajectories(*args))==1
    source.write_text('changed')
    with pytest.raises(ValueError,match='input hash'):runner.trajectories(*args)
    source.write_text('fixture');payload.write_bytes(b'changed')
    with pytest.raises(ValueError,match='trajectory hash'):runner.trajectories(*args)


def test_checkpoint_hash_and_configuration_guards(monkeypatch):
    import breakpoint_detection.paper2021_runner as runner
    monkeypatch.setitem(SETTINGS['tscp2'],'checkpoint_sha256','invalid')
    with pytest.raises(ValueError,match='checkpoint hash'):runner.validate_settings()


def test_official_output_guard_and_cli_rejects_old_protocol():
    from breakpoint_detection.cli import main,parser
    with pytest.raises(ValueError,match='official'):main(['unused','--output','results/yahoo-a4'])
    with pytest.raises(SystemExit):parser().parse_args(['unused','--protocol','2024-paper-yahoo'])


