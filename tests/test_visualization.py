"""Tests for the data-schema visualization module.

Skipped automatically if matplotlib/networkx are not installed, so the core test suite
stays dependency-free. Uses the Agg backend so it runs headless.
"""
import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("matplotlib")
pytest.importorskip("networkx")

from graphrag import visualization as viz  # noqa: E402


def _close(fig):
    import matplotlib.pyplot as plt
    assert fig is not None
    plt.close(fig)


def test_plot_ontology_returns_figure():
    _close(viz.plot_ontology())


def test_plot_knowledge_graph(graph):
    _close(viz.plot_knowledge_graph(graph, max_nodes=50))


def test_plot_knowledge_graph_with_telemetry(graph):
    # exclude_labels=[] keeps everything, including telemetry leaves
    _close(viz.plot_knowledge_graph(graph, exclude_labels=[], max_nodes=80))


def test_plot_asset_topology(graph):
    _close(viz.plot_asset_topology(graph))


def test_plot_schema_distributions(dataset):
    _close(viz.plot_schema_distributions(dataset))


def test_plot_text_overview(corpus):
    _close(viz.plot_text_overview(corpus))


def test_save_all_figures(graph, dataset, corpus, tmp_path):
    paths = viz.save_all_figures(graph, dataset, corpus, tmp_path)
    assert len(paths) == 5
    for p in paths:
        assert os.path.getsize(p) > 0
