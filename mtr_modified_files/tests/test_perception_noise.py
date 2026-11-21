import sys, os, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("pn", os.path.join(os.path.dirname(__file__), "../mtr/datasets/waymo/perception_noise.py"))
pn = importlib.util.module_from_spec(spec); spec.loader.exec_module(pn)

def make_scene(N=30, T=11, seed=0):
    r = np.random.default_rng(seed)
    x = np.zeros((N, T, 10), np.float32)
    start = r.uniform(-60, 60, (N, 2)); v = r.uniform(-10, 10, (N, 2))
    t = np.arange(T) * 0.1
    x[:, :, 0:2] = start[:, None] + v[:, None] * t[None, :, None]
    x[:, :, 3:6] = [4.5, 2.0, 1.6]; x[:, :, 6] = np.arctan2(v[:, 1], v[:, 0])[:, None]
    x[:, :, 7:9] = v[:, None]; x[:, :, 9] = 1
    x[0, :, 0:2] = 0; x[0, :, 7:9] = 0  # SDC at origin
    types = np.array(['TYPE_VEHICLE'] * 20 + ['TYPE_PEDESTRIAN'] * 10)
    return x, types

def test_all():
    x, types = make_scene()
    protect = np.zeros(len(types), bool); protect[[3, 7, 25]] = True
    noise = pn.PerceptionNoise({'LEVEL': 1.0, 'IDSWAP_PROB': 0}, seed=1)
    y, info = noise(x, types, sdc_index=0, protect_mask=protect)
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.allclose(y[0], x[0]), "SDC must stay clean"
    assert (y[protect, -1, 9] == 1).all(), "predicted agents must stay valid at current frame"
    assert not np.allclose(y[1:, :, :2], x[1:, :, :2]), "positions should be perturbed"
    # level 0 -> identity
    y0, _ = pn.PerceptionNoise({'LEVEL': 0.0}, seed=1)(x, types, 0)
    assert np.allclose(y0, x)
    # error grows with level
    errs = []
    for L in [0.5, 1, 2, 4]:
        e = [np.abs(pn.PerceptionNoise({'LEVEL': L, 'IDSWAP_PROB': 0}, seed=s)(x, types, 0)[0][1:, :, :2] - x[1:, :, :2]).mean() for s in range(20)]
        errs.append(np.mean(e))
    assert all(a < b for a, b in zip(errs, errs[1:])), errs
    # determinism with seed
    a, _ = pn.PerceptionNoise({}, seed=5)(x, types, 0); b, _ = pn.PerceptionNoise({}, seed=5)(x, types, 0)
    assert np.allclose(a, b)
    # id swap happens when agents are close
    x2 = x.copy(); x2[2] = x2[1]; x2[2, :, 0] += 1.0
    hit = False
    for s in range(50):
        _, inf = pn.PerceptionNoise({'IDSWAP_PROB': 1.0, 'IDSWAP_DIST': 2.0}, seed=s)(x2, types, 0)
        if inf['idswap'] and set(inf['idswap'][:2]) == {1, 2}: hit = True; break
    assert hit, "expected id swap between agents 1 and 2"
    print("mean |pos err| per level 0.5/1/2/4:", np.round(errs, 3))
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_all()
