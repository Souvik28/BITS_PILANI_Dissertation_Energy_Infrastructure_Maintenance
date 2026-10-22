"""Synthetic energy-infrastructure dataset generator.

Produces a connected, topologically meaningful dataset: assets (platforms) containing
process trains of components wired with CONNECTED_TO edges, telemetry per component,
failure modes, SOPs that govern repairs, and work-order history. Uses only the standard
library (a tiny built-in Faker-style name generator) so it runs anywhere.

The generated graph is deliberately engineered so that questions such as
"failure modes for pumps downstream of Separator V-202" have deterministic answers.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .config import DataGenConfig
from .schema import (
    COMPONENT_TYPES,
    CRITICALITY_TIERS,
    Asset,
    Component,
    FailureMode,
    SOP,
    Telemetry,
    WorkOrder,
)

MANUFACTURERS = ["Siemens", "GE", "Baker Hughes", "Flowserve", "Sulzer", "Alfa Laval"]
LOCATIONS = ["North Sea Block 15", "Gulf Coast Terminal", "Barents Shelf", "Rotterdam LNG"]
RESOLVERS = ["A. Haugen", "M. Okafor", "L. Nielsen", "R. Patel", "S. Dutta", "C. Berg"]

# Canonical named tags so benchmark questions can reference real equipment.
# Each platform gets one separator with a well-known tag (e.g. V-202).
SEPARATOR_TAGS = ["V-202", "V-305", "V-118", "V-440"]

FAILURE_MODE_DEFS = [
    ("FM-01", "Bearing Seal Degradation", "High vibration > 4.5 mm/s", "Mechanical", 4),
    ("FM-02", "Thermal Overload", "Temperature exceeded 95 C", "Thermal", 5),
    ("FM-03", "Cavitation", "Fluctuating suction pressure and noise", "Hydraulic", 3),
    ("FM-04", "Fouling / Scaling", "Reduced flow rate and rising dP", "Process", 2),
    ("FM-05", "Seal Gas Leak", "Detected hydrocarbon leak at seal", "Integrity", 5),
    ("FM-06", "Corrosion Thinning", "Wall thickness below retirement limit", "Integrity", 4),
]

# Which failure modes are plausible for which component type (grounding signal).
TYPE_TO_FAILURES = {
    "Centrifugal_Pump": ["FM-01", "FM-03", "FM-06"],
    "Gas_Turbine": ["FM-01", "FM-02", "FM-05"],
    "Heat_Exchanger": ["FM-02", "FM-04", "FM-06"],
    "Pressure_Valve": ["FM-05", "FM-06"],
    "Separator": ["FM-04", "FM-06"],
    "Compressor": ["FM-01", "FM-02", "FM-05"],
}

ISSUE_TYPES = ["Corrective", "Preventive", "Inspection", "Breakdown"]
PRIORITIES = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]


def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


class EnergyDataGenerator:
    """Builds a coherent synthetic dataset from a config + seed (reproducible)."""

    def __init__(self, cfg: DataGenConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # ------------------------------------------------------------------ entities
    def _make_assets(self) -> List[Asset]:
        assets = []
        for i in range(self.cfg.num_assets):
            aid = f"PLATFORM_ALPHA_{i + 1}"
            assets.append(
                Asset(
                    asset_id=aid,
                    name=f"Alpha Platform {i + 1}",
                    system_code=f"SYS-{100 + i}",
                    criticality_tier=self.rng.randint(1, 4),
                    installation_date=_date_str(
                        datetime(2005 + i, self.rng.randint(1, 12), self.rng.randint(1, 28))
                    ),
                    location=self.rng.choice(LOCATIONS),
                )
            )
        return assets

    def _make_components(self, assets: List[Asset]) -> List[Component]:
        comps: List[Component] = []
        for a_idx, asset in enumerate(assets):
            # Unique per-platform prefix (P1, P2, ...). Using asset.asset_id[:3] would
            # collide because every platform is named "PLATFORM_ALPHA_n" -> "PLA".
            prefix = f"P{a_idx + 1}"
            # Guarantee one named Separator per platform for traceable questions.
            tag = SEPARATOR_TAGS[a_idx % len(SEPARATOR_TAGS)]
            sep = Component(
                comp_id=f"{prefix}_SEP_{tag}",
                asset_id=asset.asset_id,
                name=f"Separator {tag}",
                type="Separator",
                model=f"SEP-{self.rng.randint(1000, 9999)}",
                manufacturer=self.rng.choice(MANUFACTURERS),
                design_pressure_bar=round(self.rng.uniform(40, 120), 1),
                operating_temp_max_c=round(self.rng.uniform(60, 140), 1),
                installed_year=self.rng.randint(2010, 2022),
                status=self.rng.choice(["OPERATIONAL", "MAINTENANCE_REQUIRED", "DEGRADED"]),
            )
            comps.append(sep)

            for j in range(self.cfg.comps_per_asset - 1):
                ctype = self.rng.choice([c for c in COMPONENT_TYPES if c != "Separator"])
                comps.append(
                    Component(
                        comp_id=f"{prefix}_EQ_{100 + j}",
                        asset_id=asset.asset_id,
                        name=f"{ctype.replace('_', ' ')} {prefix}-{100 + j}",
                        type=ctype,
                        model=f"{ctype[:3].upper()}-{self.rng.randint(1000, 9999)}",
                        manufacturer=self.rng.choice(MANUFACTURERS),
                        design_pressure_bar=round(self.rng.uniform(10, 150), 1),
                        operating_temp_max_c=round(self.rng.uniform(40, 160), 1),
                        installed_year=self.rng.randint(2010, 2022),
                        status=self.rng.choice(
                            ["OPERATIONAL", "MAINTENANCE_REQUIRED", "DEGRADED"]
                        ),
                    )
                )
        return comps

    def _make_connections(self, comps: List[Component]) -> List[Tuple[str, str]]:
        """Wire a process flow per platform: Separator -> pumps/exchangers downstream.

        Returns (source, target) pairs meaning source CONNECTED_TO target (flow dir).
        """
        edges: List[Tuple[str, str]] = []
        by_asset: Dict[str, List[Component]] = {}
        for c in comps:
            by_asset.setdefault(c.asset_id, []).append(c)

        for asset_id, members in by_asset.items():
            separators = [c for c in members if c.type == "Separator"]
            others = [c for c in members if c.type != "Separator"]
            if not separators or not others:
                continue
            sep = separators[0]
            self.rng.shuffle(others)
            # Separator feeds the first ~half of the components (downstream),
            # which then chain to the rest. This creates multi-hop topology.
            split = max(1, len(others) // 2)
            downstream_direct = others[:split]
            downstream_second = others[split:]
            for c in downstream_direct:
                edges.append((sep.comp_id, c.comp_id))
            # chain second tier off the first tier
            for idx, c in enumerate(downstream_second):
                parent = downstream_direct[idx % len(downstream_direct)]
                edges.append((parent.comp_id, c.comp_id))
        return edges

    def _make_failure_modes(self) -> List[FailureMode]:
        return [
            FailureMode(
                failure_id=code,
                failure_code=code,
                name=name,
                symptom=symptom,
                root_cause_category=cat,
                severity_index=sev,
            )
            for code, name, symptom, cat, sev in FAILURE_MODE_DEFS
        ]

    def _make_sops(self, failure_modes: List[FailureMode]) -> List[SOP]:
        sops: List[SOP] = []
        for i, fm in enumerate(failure_modes):
            sid = f"SOP-{i + 1:02d}"
            target_types = [
                t for t, codes in TYPE_TO_FAILURES.items() if fm.failure_code in codes
            ]
            target = self.rng.choice(target_types) if target_types else "General"
            body = (
                f"Standard Operating Procedure {sid} (rev B) governs the repair of "
                f"failure mode {fm.failure_code} '{fm.name}'. Applicable to "
                f"{target.replace('_', ' ')} equipment. Symptom signature: {fm.symptom}. "
                f"Isolate and depressurise the component, apply lockout-tagout, verify zero "
                f"energy state, then inspect and replace the affected subassembly. "
                f"Hazard class {['A', 'B', 'C'][i % 3]} requires a confined-space permit "
                f"and gas test before entry. Recommission under 80% load and record "
                f"vibration and temperature baselines."
            )
            sops.append(
                SOP(
                    sop_id=sid,
                    failure_id=fm.failure_id,
                    title=f"Repair procedure for {fm.name}",
                    revision="B",
                    target_component_type=target,
                    hazard_class=["A", "B", "C"][i % 3],
                    approved_year=self.rng.randint(2018, 2024),
                    body=body,
                )
            )
        return sops

    def _make_work_orders(
        self, comps: List[Component], failure_modes: List[FailureMode]
    ) -> List[WorkOrder]:
        fm_by_code = {fm.failure_code: fm for fm in failure_modes}
        orders: List[WorkOrder] = []
        for i in range(self.cfg.work_orders):
            comp = self.rng.choice(comps)
            eligible = TYPE_TO_FAILURES.get(comp.type, [fm.failure_code for fm in failure_modes])
            fm = fm_by_code[self.rng.choice(eligible)]
            date = datetime.now() - timedelta(days=self.rng.randint(1, 730))
            orders.append(
                WorkOrder(
                    wo_id=f"WO-{202400 + i}",
                    comp_id=comp.comp_id,
                    failure_id=fm.failure_id,
                    creation_date=_date_str(date),
                    issue_type=self.rng.choice(ISSUE_TYPES),
                    priority=self.rng.choice(PRIORITIES),
                    cost_usd=round(self.rng.uniform(1_200, 85_000), 2),
                    duration_hours=round(self.rng.uniform(2, 96), 1),
                    resolved_by=self.rng.choice(RESOLVERS),
                    summary=(
                        f"Reported {fm.symptom.lower()} on {comp.name} ({comp.comp_id}). "
                        f"Classified as failure mode {fm.failure_code} ({fm.name})."
                    ),
                    resolution=(
                        f"Applied procedure for {fm.name}. Replaced affected subassembly on "
                        f"{comp.name} and recalibrated. Tested under 80% load; vibration and "
                        f"temperature within limits."
                    ),
                )
            )
        return orders

    def _make_telemetry(self, comps: List[Component]) -> List[Telemetry]:
        telem: List[Telemetry] = []
        counter = 0
        for comp in comps:
            for _ in range(self.cfg.telemetry_per_comp):
                ts = datetime.now() - timedelta(hours=self.rng.randint(1, 24 * 60))
                # DEGRADED components trend toward higher vibration / temp.
                vib_base = 5.5 if comp.status == "DEGRADED" else 2.2
                telem.append(
                    Telemetry(
                        metric_id=f"TM-{counter:06d}",
                        comp_id=comp.comp_id,
                        timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        vibration_mm_s=round(abs(self.rng.gauss(vib_base, 1.1)), 2),
                        temp_celsius=round(self.rng.gauss(70, 15), 1),
                        pressure_bar=round(abs(self.rng.gauss(comp.design_pressure_bar * 0.7, 8)), 1),
                        flow_rate_m3h=round(abs(self.rng.gauss(120, 30)), 1),
                    )
                )
                counter += 1
        return telem

    # ------------------------------------------------------------------ public API
    def generate(self) -> Dict:
        assets = self._make_assets()
        components = self._make_components(assets)
        connections = self._make_connections(components)
        failure_modes = self._make_failure_modes()
        sops = self._make_sops(failure_modes)
        work_orders = self._make_work_orders(components, failure_modes)
        telemetry = self._make_telemetry(components)

        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "seed": self.cfg.seed,
                "counts": {
                    "assets": len(assets),
                    "components": len(components),
                    "connections": len(connections),
                    "failure_modes": len(failure_modes),
                    "sops": len(sops),
                    "work_orders": len(work_orders),
                    "telemetry": len(telemetry),
                },
            },
            "assets": [a.as_props() for a in assets],
            "components": [c.as_props() for c in components],
            "connections": [{"source": s, "target": t} for s, t in connections],
            "failure_modes": [f.as_props() for f in failure_modes],
            "sops": [s.as_props() for s in sops],
            "work_orders": [w.as_props() for w in work_orders],
            "telemetry": [t.as_props() for t in telemetry],
        }


def generate_dataset(cfg: DataGenConfig) -> Dict:
    return EnergyDataGenerator(cfg).generate()
