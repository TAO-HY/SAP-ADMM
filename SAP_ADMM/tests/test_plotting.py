"""Plot-only regression checks: styling must not change saved measurements."""
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments import plot_general_a, plot_mnist, plot_signal


@pytest.fixture
def figures(monkeypatch):
    captured = {}

    def capture(fig, folder, basename, **kwargs):
        captured[basename] = fig

    for module in (plot_signal, plot_mnist, plot_general_a):
        monkeypatch.setattr(module, "save_figure", capture)
    yield captured
    for fig in captured.values():
        plt.close(fig)


@pytest.mark.parametrize("trials", [1, 4])
def test_signal_plot_preserves_means_categories_and_recovery(tmp_path, figures, trials):
    f1 = np.array([[[1.0, 0.9999999], [0.9, 0.8]],
                   [[0.8, 0.9], [0.7, 1.0]],
                   [[0.7, 0.8], [1.0, 0.7]],
                   [[0.9, 0.7], [0.8, 0.9]]])[:trials]
    seconds = np.arange(1, trials * 4 + 1, dtype=float).reshape(trials, 2, 2)
    methods = np.array(["sap_admm", "sap_admm_halpern"])
    probabilities = np.array([0.0, 0.2])
    np.savez(tmp_path / "results.npz", probabilities=probabilities, methods=methods,
             f1=f1, seconds=seconds, completed=np.ones_like(f1, dtype=bool))
    truth = np.arange(10, dtype=float)
    observations = np.stack([truth + 0.1, truth + 0.2])
    recovered = np.tile(truth, (2, 2, 1))
    np.savez(tmp_path / "recovery_example.npz", truth=truth, observations=observations,
             recovered=recovered, probabilities=probabilities, methods=methods)
    files = list(tmp_path.glob("*.npz"))
    before = [hashlib.sha256(path.read_bytes()).digest() for path in files]

    plot_signal.plot(tmp_path)

    mean_f1, mean_time = f1.mean(0), seconds.mean(0)
    for ax, expected in zip(figures["signal_metrics"].axes,
                            (mean_f1, mean_time, mean_f1 / mean_time)):
        for k, line in enumerate(ax.lines):
            np.testing.assert_array_equal(line.get_xdata(), probabilities)
            np.testing.assert_allclose(line.get_ydata(), expected[:, k])
    assert len(figures["signal_metrics"].axes[0].collections) == 2
    for k, ax in enumerate(figures["signal_f1_distribution"].axes):
        values = f1[:, :, k]
        expected = np.stack([(values == 1).sum(0),
                             ((values >= 0.9) & (values < 1)).sum(0),
                             ((values >= 0.8) & (values < 0.9)).sum(0),
                             (values < 0.8).sum(0)])
        heights = np.array([bar.get_height() for bar in ax.patches]).reshape(4, 2)
        np.testing.assert_array_equal(heights, expected)
        np.testing.assert_array_equal(heights.sum(0), np.full(2, trials))
    axes = figures["signal_recovery"].axes
    assert len(axes) == 6
    for row in range(2):
        np.testing.assert_array_equal(axes[3 * row].lines[0].get_ydata(), observations[row])
        for k in range(2):
            np.testing.assert_array_equal(axes[3 * row + k + 1].lines[0].get_ydata(), truth)
            np.testing.assert_array_equal(axes[3 * row + k + 1].lines[1].get_ydata(), recovered[row, k])
    assert before == [hashlib.sha256(path.read_bytes()).digest() for path in files]


@pytest.mark.parametrize("count", [1, 10])
def test_mnist_square_tiles_preserve_images(tmp_path, figures, count):
    clean = np.linspace(0, 1, count * 28 * 28).reshape(count, 28, 28)
    observations, recovered = clean * 0.8, clean * 0.9
    np.savez(tmp_path / "results.npz", digits=np.arange(count), clean=clean,
             observations=observations[None], recovered=recovered[None],
             completed=np.ones((1, count), dtype=bool))
    plot_mnist.plot(tmp_path)
    fig = figures["mnist_restoration"]
    assert len(fig.axes) == 3 * count
    for row, images in enumerate((clean, observations, recovered)):
        for col in range(count):
            ax = fig.axes[row * count + col]
            np.testing.assert_array_equal(ax.images[0].get_array(), images[col])
            bounds = ax.get_position()
            assert bounds.width * fig.get_figwidth() == pytest.approx(
                bounds.height * fig.get_figheight())


def test_general_a_panel_order_and_means(tmp_path, figures):
    x = np.array([200, 300, 400, 500, 600])
    arrays = {"f1": np.ones((2, 5)), "iterations": np.full((2, 5), 8239.95),
              "mse": np.array([[0.003] * 5, [0.005] * 5])}
    np.savez(tmp_path / "results.npz", Nmax_list=x, **arrays,
             completed=np.ones((2, 5), dtype=bool))
    plot_general_a.plot(tmp_path)
    for ax, key in zip(figures["general_A_sensitivity"].axes, ("f1", "iterations", "mse")):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), arrays[key].mean(0))
        lower, upper = ax.get_ylim()
        assert lower < arrays[key].mean() < upper


@pytest.mark.parametrize("module", [plot_signal, plot_mnist, plot_general_a])
def test_incomplete_results_are_not_plotted(tmp_path, figures, module):
    np.savez(tmp_path / "results.npz", completed=np.array([False]))
    with pytest.raises(ValueError, match="incomplete"):
        module.plot(tmp_path)
    assert not figures
