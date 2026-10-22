"""Pytest bootstrap: make ``src`` importable and provide shared fixtures."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest  # noqa: E402

from graphrag.config import DataGenConfig  # noqa: E402
from graphrag.data_generation import generate_dataset  # noqa: E402
from graphrag.graph_store import NetworkXGraphStore  # noqa: E402
from graphrag.ingest import build_corpus, build_graph  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg():
    return DataGenConfig(
        seed=7, num_assets=2, comps_per_asset=6, work_orders=40, telemetry_per_comp=3
    )


@pytest.fixture(scope="session")
def dataset(small_cfg):
    return generate_dataset(small_cfg)


@pytest.fixture()
def graph(dataset):
    store = NetworkXGraphStore()
    build_graph(store, dataset)
    return store


@pytest.fixture()
def corpus(dataset):
    return build_corpus(dataset)
