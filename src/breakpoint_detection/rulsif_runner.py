"""Compatibility entrypoint for the fixed rulsif baseline."""
import argparse
from .protocol import SETTINGS

def configuration(args=None, margin=None):
    config = dict(SETTINGS['rulsif'])
    if margin is not None:
        config.update(config['by_margin'][str(margin)])
    return config

def run(args):
    from .cli import parser
    from .paper2021_runner import run as evaluate
    defaults = vars(parser().parse_args([str(args.input)]))
    defaults.update(vars(args))
    defaults['model'] = 'rulsif'
    return evaluate(argparse.Namespace(**defaults))
