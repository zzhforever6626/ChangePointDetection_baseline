"""The training entry prepares its own inputs without held-out training pairs."""
import importlib.util
from pathlib import Path
import numpy as np


def test_training_entry_uses_repository_data_and_only_first_forty_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('train_entry', root / 'scripts/train_tscp2.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    config = entry.prepare(tmp_path)
    assert config['seed'] == 0 and config['max_epochs'] == 1600
    assert (config['window'], config['batch_size'], config['code_size'], config['learning_rate']) == (32, 32, 16, .0003)
    assert 'checkpoint' not in config and 'by_margin' not in config
    with np.load(tmp_path / 'development-input.npz', allow_pickle=False) as data:
        assert data['values'].shape == (60, 1680)
        np.testing.assert_array_equal(data['train_stops'], [1680] * 40 + [0] * 20)
        _, locations = entry.training_pairs(data['values'], data['train_stops'], config['window'],
                                            config['training_stride'], last_start_inclusive=False)
    assert len(locations) == 12960 and set(locations[:, 0]) == set(range(40))
    batches = entry.separated_batches(locations, config['batch_size'], 2 * config['window'], config['seed'])
    assert batches.shape == (404, 32)
    for batch in batches:
        sites = locations[batch]
        for i, left in enumerate(sites):
            assert all(left[0] != right[0] or abs(left[1] - right[1]) >= 64 for right in sites[i+1:])
