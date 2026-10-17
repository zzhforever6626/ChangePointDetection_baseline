"""Label-free heldout scoring of a validation-selected TSCP2 checkpoint."""
import argparse
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('tscp2_core',Path(__file__).parent/'models/tscp2.py')
core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
import numpy as np

def score(input_path,checkpoint_path,output,device_name):
    checkpoint=core.torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config,window=checkpoint['parameters'],checkpoint['window']
    # A checkpoint is part of the fixed baseline, not an implicit config override.
    import json
    settings=json.loads((Path(__file__).parent/'baseline_settings.json').read_text(encoding='utf-8'))['tscp2']
    for row in settings['by_margin'].values():
        if row['window'] != window:
            raise ValueError('checkpoint window differs from fixed settings')
        for key,value in settings.items():
            if key in config and key not in ('fixed_claim','adapters') and config[key] != value:
                raise ValueError(f'checkpoint parameter differs from fixed settings: {key}')
        for key in ('batch_size','code_size','learning_rate'):
            if config[key] != row[key]:
                raise ValueError(f'checkpoint parameter differs from fixed settings: {key}')
    device=core.initialize(config['seed'],device_name)
    model=core.TSCP2Encoder(window,config).to(device)
    model.load_state_dict(checkpoint['state_dict'],strict=True); model.eval()
    with np.load(input_path,allow_pickle=False) as data: values=data['values']
    step=config['inference_stride']; similarity=core.score_encoder(model,values,window,device,step)
    scores=np.array([core._adapter.similarity_drop(row,window,step) for row in similarity])
    peaks=np.array([core._adapter.phase_peak_scores(row,window,len(row),step) for row in scores])
    available=np.where(np.isfinite(scores),np.arange(scores.shape[1])[None,:]+window,-1)
    np.savez_compressed(output,similarity=similarity,scores=scores,peaks=peaks,available=available)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('input','checkpoint','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    args=parser.parse_args(); score(args.input,args.checkpoint,args.output,args.device)
