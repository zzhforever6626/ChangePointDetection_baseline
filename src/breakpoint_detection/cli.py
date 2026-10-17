"""Fixed validation-selected baseline evaluation."""
import argparse
from pathlib import Path
from .protocol import MARGINS
PAPER_MARGINS=MARGINS

def parser():
    result = argparse.ArgumentParser(description='Evaluate fixed FLOSS, RuLSIF and TSCP2 baselines')
    result.add_argument('input', type=Path)
    result.add_argument('--model', choices=('all','floss','floss-author','rulsif','tscp2'), default='all')
    result.add_argument('--protocol', choices=('tscp2-2021-yahoo',), default='tscp2-2021-yahoo')
    result.add_argument('--output', type=Path, default=Path('experiments/runs/fixed-baselines'))
    result.add_argument('--workers', type=int, default=3)
    result.add_argument('--tscp2-python', type=Path, default=None)
    result.add_argument('--tscp2-device', choices=('cpu','cuda','auto'), default='cpu')
    result.add_argument('--resume', action='store_true', help='verify a completed evaluation without rerunning it')
    result.add_argument('--reuse-scores', type=Path, default=None,
                        help='reuse fixed traditional trajectories after checking settings, inputs, source and hashes')
    return result


def main(argv=None):
    from .paper2021_runner import run
    args = parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError('workers must be positive')
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
