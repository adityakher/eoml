import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from eoml import viz


def make_map(values):
    a = np.asarray(values)
    return xr.DataArray(
        a, dims=("y", "x"),
        coords={"y": np.arange(a.shape[0]), "x": np.arange(a.shape[1])},
    )


def test_plot_confidence_histogram_runs():
    fig = viz.plot_confidence_histogram(
        np.random.rand(500), make_map(np.random.rand(5, 6)), threshold=0.5
    )
    assert fig is not None
    plt.close(fig)


def test_plot_confidence_histogram_handles_nan_and_plain_arrays():
    ref = np.array([0.9, 0.8, np.nan, 0.5])  # NaN must be dropped, not crash
    scn = np.array([0.1, 0.2, 0.3])
    fig = viz.plot_confidence_histogram(ref, scn)
    plt.close(fig)


def test_plot_confidence_map_runs():
    fig = viz.plot_confidence_map(make_map(np.random.rand(5, 6)))
    assert fig is not None
    plt.close(fig)


def test_plot_class_map_without_confidence_has_no_patches():
    classes = make_map(np.zeros((4, 5), dtype=int))
    fig = viz.plot_class_map(classes)
    assert len(fig.axes[0].patches) == 0
    plt.close(fig)


def test_plot_class_map_hatches_exactly_the_low_confidence_chips():
    classes = make_map(np.zeros((4, 5), dtype=int))
    conf = make_map(np.linspace(0.0, 1.0, 20).reshape(4, 5))
    threshold = 0.5
    fig = viz.plot_class_map(classes, confidence=conf, low_conf_threshold=threshold)
    n_low = int((conf.values < threshold).sum())
    # one hatched Rectangle per low-confidence chip (legend uses a proxy handle,
    # so it does not add to ax.patches)
    assert len(fig.axes[0].patches) == n_low
    plt.close(fig)


def test_plot_class_map_rejects_mismatched_confidence():
    classes = make_map(np.zeros((4, 5), dtype=int))
    conf = make_map(np.zeros((4, 6)))  # wrong width
    try:
        viz.plot_class_map(classes, confidence=conf)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_plot_ndvi_class_scatter_runs():
    class_map = make_map(np.array([[0, 1, 2], [3, 0, 1]]))
    ndvi_mean = make_map(np.random.rand(2, 3))
    ndvi_std = make_map(np.random.rand(2, 3) * 0.2)
    fig = viz.plot_ndvi_class_scatter(class_map, ndvi_mean, ndvi_std)
    assert fig is not None
    plt.close(fig)
