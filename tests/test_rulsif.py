"""Independent scalar MATLAB-formula checks for the source RuLSIF port."""
import numpy as np
import pytest

from breakpoint_detection.models.author_rulsif import fit_direction, score_series, paired_samples


def scalar_reference(numerator, denominator):
    rng = np.random.RandomState(0)
    centers = numerator[rng.permutation(20)]
    samples = np.concatenate((denominator,numerator))
    distances = [sum((samples[i]-samples[j])**2) for i in range(40) for j in range(i+1,40)]
    base = np.sqrt(.5*np.median([d for d in distances if d>0]))
    risks = []
    kernels = []
    for multiplier in (.6,.8,1.,1.2,1.4):
        sigma = base*multiplier
        nu = np.array([[np.exp(-sum((x-c)**2)/(2*sigma**2)) for c in centers] for x in numerator])
        de = np.array([[np.exp(-sum((x-c)**2)/(2*sigma**2)) for c in centers] for x in denominator])
        nu_order,de_order = rng.permutation(20),rng.permutation(20)
        risk = 0.
        for fold in range(5):
            held = np.arange(fold*4,(fold+1)*4)
            train = np.setdiff1d(np.arange(20),held)
            a,b = nu[nu_order[train]],de[de_order[train]]
            H = .1*(a.T@a)/16+.9*(b.T@b)/16
            theta = np.linalg.solve(H+.01*np.eye(20),a.mean(axis=0))
            g_nu,g_de = nu[nu_order[held]]@theta,de[de_order[held]]@theta
            risk += .05*np.mean(g_nu**2)+.45*np.mean(g_de**2)-np.mean(g_nu)
        risks.append(risk/5)
        kernels.append((nu,de,sigma))
    nu,de,sigma = kernels[int(np.argmin(risks))]
    theta = np.linalg.solve(.1*(nu.T@nu)/20+.9*(de.T@de)/20+.01*np.eye(20),nu.mean(axis=0))
    g_nu,g_de = nu@theta,de@theta
    pe = np.mean(g_nu)-.05*np.mean(g_nu**2)-.45*np.mean(g_de**2)-.5
    return pe,sigma,np.asarray(risks),theta


@pytest.mark.parametrize("seed", [2,7,19])
def test_direction_matches_independent_five_fold_relative_risk(seed):
    rng = np.random.default_rng(seed)
    nu,de = rng.normal(size=(20,10)),rng.normal(.5,size=(20,10))
    expected = scalar_reference(nu,de)
    actual = fit_direction(nu,de)
    assert np.min(expected[3]) < 0  # This fixture detects coefficient clipping.
    np.testing.assert_allclose(actual[0],expected[0],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(actual[1],expected[1],rtol=1e-12)
    np.testing.assert_allclose(actual[2],expected[2],rtol=1e-10,atol=1e-12)


def test_full_reverse_scan_and_literal_plot_geometry():
    values = np.random.default_rng(41).normal(size=54)
    boundary, _, _, available, plot = score_series(values, 10, n=20, alpha=.1)
    assert np.flatnonzero(np.isfinite(boundary)).tolist() == list(range(20,26))
    for start in (0,2,5):
        scores = []
        for observation,offset in ((values,start),(values[::-1],5-start)):
            vectors = np.lib.stride_tricks.sliding_window_view(observation,10)[offset:offset+40].copy()
            vectors /= np.std(vectors,axis=0,ddof=1)
            scores.append(scalar_reference(vectors[20:],vectors[:20])[0])
        np.testing.assert_allclose(boundary[20+start],sum(scores),rtol=1e-9,atol=1e-11)
        assert plot[start] == 48+start
        assert available[20+start] == 49+start


def test_source_invalid_constant_buffer_is_not_silently_regularized():
    with pytest.raises(ValueError,match="zero coordinate std"):
        score_series(np.ones(60), 10, n=20, alpha=.01)
    with pytest.raises(ValueError,match="Incomplete author buffer"):
        paired_samples(np.arange(48.), 0, 10, n=20)
