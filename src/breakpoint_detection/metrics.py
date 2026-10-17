from __future__ import annotations

import numpy as np


def paper_event_metrics(truth, prediction, margin):
    """2021 event counts; overlapping margins use a closest-edge-first adapter.

    Equal-coordinate predictions remain separate; no intervals or FP runs merge.
    """
    if margin <= 0:
        raise ValueError('margin must be positive')
    truth = np.sort(np.asarray(truth, dtype=int))
    prediction = np.sort(np.asarray(prediction, dtype=int))
    edges = sorted((abs(int(p)-int(t)), int(p), int(t), i, j)
                   for i, p in enumerate(prediction) for j, t in enumerate(truth)
                   if abs(int(p)-int(t)) < margin)
    used_p, used_t, errors = set(), set(), []
    for distance, _, _, i, j in edges:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); errors.append(distance)
    tp = len(errors)
    fp, fn = len(prediction)-tp, len(truth)-tp
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp/len(prediction) if len(prediction) else 0.,
                recall=tp/len(truth) if len(truth) else 0.,
                f1=2*tp/(len(prediction)+len(truth)) if len(prediction)+len(truth) else 0.,
                mean_delay=float(np.mean(errors)) if errors else float('nan'))
