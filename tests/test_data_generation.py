"""Tests for synthetic data generation."""


def test_counts_match_config(dataset, small_cfg):
    counts = dataset["metadata"]["counts"]
    assert counts["assets"] == small_cfg.num_assets
    assert counts["components"] == small_cfg.num_assets * small_cfg.comps_per_asset
    assert counts["work_orders"] == small_cfg.work_orders
    assert counts["telemetry"] == counts["components"] * small_cfg.telemetry_per_comp


def test_reproducible(small_cfg):
    from graphrag.data_generation import generate_dataset

    a = generate_dataset(small_cfg)
    b = generate_dataset(small_cfg)
    assert a["components"] == b["components"]
    assert a["work_orders"] == b["work_orders"]


def test_every_platform_has_named_separator(dataset):
    seps = [c for c in dataset["components"] if c["type"] == "Separator"]
    # one separator per asset, each with a V-### style tag in the name
    assets = {c["asset_id"] for c in seps}
    assert len(assets) == dataset["metadata"]["counts"]["assets"]
    assert all("V-" in c["name"] for c in seps)


def test_work_orders_reference_valid_entities(dataset):
    comp_ids = {c["comp_id"] for c in dataset["components"]}
    fm_ids = {f["failure_id"] for f in dataset["failure_modes"]}
    for w in dataset["work_orders"]:
        assert w["comp_id"] in comp_ids
        assert w["failure_id"] in fm_ids


def test_connections_form_flow(dataset):
    comp_ids = {c["comp_id"] for c in dataset["components"]}
    assert dataset["connections"], "expected at least one CONNECTED_TO edge"
    for conn in dataset["connections"]:
        assert conn["source"] in comp_ids
        assert conn["target"] in comp_ids
