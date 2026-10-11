from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

SEED = 42

PRIORITY_LABELS = [(75, "Критический"), (50, "Высокий"), (25, "Средний"), (0, "Низкий")]
PRIORITY_WEIGHTS = {"economic": 0.4, "urgency": 0.35, "criticality": 0.25}


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _priority(economic: float, urgency: float, criticality: float) -> tuple[int, str]:
    score = round(
        PRIORITY_WEIGHTS["economic"] * economic
        + PRIORITY_WEIGHTS["urgency"] * urgency
        + PRIORITY_WEIGHTS["criticality"] * criticality
    )
    label = next(label for threshold, label in PRIORITY_LABELS if score >= threshold)
    return score, label


def _as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, datetime.min.time())


def generate_domovoy_data(as_of: date | None = None) -> dict[str, pd.DataFrame]:
    """Generate deterministic synthetic data for the Domovoy demo according to the v2 data contract."""
    as_of = _week_start(as_of or date.today())
    rng = np.random.default_rng(SEED)

    business_units = pd.DataFrame(
        [
            {"bu_id": "BU-CENTER", "name": "Центр", "bu_type": "trade", "stores_count": 120, "default_dc_id": "DC-MSK", "soe_meeting_day": "Mon", "fast_track_enabled": False},
            {"bu_id": "BU-REGIONS", "name": "Регионы", "bu_type": "trade", "stores_count": 210, "default_dc_id": "DC-EKB", "soe_meeting_day": "Tue", "fast_track_enabled": False},
            {"bu_id": "BU-ONLINE", "name": "Онлайн", "bu_type": "trade", "stores_count": 0, "default_dc_id": "DC-MSK", "soe_meeting_day": "Mon", "fast_track_enabled": True},
            {"bu_id": "BU-IMPORT", "name": "Импорт", "bu_type": "import", "stores_count": 0, "default_dc_id": "DC-MSK", "soe_meeting_day": "Tue", "fast_track_enabled": True},
        ]
    )

    distribution_centers = pd.DataFrame(
        [
            {"dc_id": "DC-MSK", "name": "РЦ Москва", "inbound_capacity_containers_week": 40, "transfer_days_between_dc": 4},
            {"dc_id": "DC-EKB", "name": "РЦ Екатеринбург", "inbound_capacity_containers_week": 20, "transfer_days_between_dc": 4},
        ]
    )

    categories = pd.DataFrame(
        [
            {"category_id": "CAT-DISH", "name": "Посуда и кухонные принадлежности", "demand_profile": "stable", "allocation_rule": "pro_rata", "season_start": None, "season_end": None, "markdown_rate_after_season": 0.0, "event_detection_mode": "corridor"},
            {"category_id": "CAT-SDA", "name": "МБТ и мелкая бытовая техника", "demand_profile": "promo", "allocation_rule": "pro_rata", "season_start": None, "season_end": None, "markdown_rate_after_season": 0.0, "event_detection_mode": "corridor"},
            {"category_id": "CAT-DECOR", "name": "Декор и праздничные товары", "demand_profile": "seasonal", "allocation_rule": "sell_through", "season_start": "2026-10-01", "season_end": "2026-12-31", "markdown_rate_after_season": 0.5, "event_detection_mode": "sell_through_curve"},
        ]
    )

    suppliers = pd.DataFrame(
        [
            {"supplier_id": "SUP-CN-01", "country": "Китай", "production_weeks_min": 6, "production_weeks_max": 8, "transit_mode_default": "sea", "transit_weeks_min": 5, "transit_weeks_max": 7, "historical_otif": 0.86, "moq_units": 500},
            {"supplier_id": "SUP-CN-02", "country": "Китай", "production_weeks_min": 6, "production_weeks_max": 8, "transit_mode_default": "rail", "transit_weeks_min": 5, "transit_weeks_max": 7, "historical_otif": 0.87, "moq_units": 500},
            {"supplier_id": "SUP-VN-01", "country": "Вьетнам", "production_weeks_min": 7, "production_weeks_max": 9, "transit_mode_default": "sea", "transit_weeks_min": 5, "transit_weeks_max": 7, "historical_otif": 0.84, "moq_units": 600},
            {"supplier_id": "SUP-VN-02", "country": "Вьетнам", "production_weeks_min": 7, "production_weeks_max": 9, "transit_mode_default": "rail", "transit_weeks_min": 5, "transit_weeks_max": 7, "historical_otif": 0.82, "moq_units": 600},
            {"supplier_id": "SUP-TR-01", "country": "Турция", "production_weeks_min": 4, "production_weeks_max": 6, "transit_mode_default": "sea", "transit_weeks_min": 3, "transit_weeks_max": 4, "historical_otif": 0.9, "moq_units": 400},
            {"supplier_id": "SUP-TR-02", "country": "Турция", "production_weeks_min": 4, "production_weeks_max": 6, "transit_mode_default": "rail", "transit_weeks_min": 3, "transit_weeks_max": 4, "historical_otif": 0.88, "moq_units": 400},
        ]
    )

    sku_catalog = {
        "CAT-DISH": [
            "Плита индукционная",
            "Набор кастрюль",
            "Блендер погружной",
            "Чайник электрический",
            "Сковорода антипригарная",
            "Кофемолка",
            "Терка металлическая",
            "Электрический гриль",
            "Термос 1 л",
            "Соковыжималка",
            "Сервиз столовый",
            "Набор ножей",
        ],
        "CAT-SDA": [
            "Пылесос компактный",
            "Утюг паровой",
            "Швейная машинка",
            "Фен профессиональный",
            "Миксер электрический",
            "Плойка для волос",
            "Вентилятор напольный",
            "Пароочиститель",
        ],
        "CAT-DECOR": [
            "Свеча керамическая",
            "Новогодний венок",
            "Подушка декоративная",
            "Плед в клетку",
            "Шарики для праздника",
            "Лампа настольная",
            "Ковер минимализм",
            "Табурет тканевый",
            "Шторы светлые",
            "Панно настенное",
        ],
    }

    supplier_cycle = suppliers["supplier_id"].tolist()
    skus = []
    sku_codes = []
    for category_id, names in sku_catalog.items():
        for idx, name in enumerate(names, start=1):
            supplier_id = supplier_cycle[(len(skus) + idx) % len(supplier_cycle)]
            base_cost = 1200 if category_id == "CAT-DISH" else 900 if category_id == "CAT-SDA" else 350
            unit_cost = int(base_cost + rng.integers(80, 1800) if category_id != "CAT-DECOR" else rng.integers(25, 450))
            sku = f"SKU-{category_id[-3:]}-{idx:03d}" if category_id != "CAT-DECOR" else f"SKU-DEC-{idx:03d}"
            skus.append(
                {
                    "sku": sku,
                    "name": name,
                    "category_id": category_id,
                    "supplier_id": supplier_id,
                    "abc_class": "A" if idx <= 3 else "B" if idx <= 7 else "C",
                    "unit_cost": unit_cost,
                    "import_markup": 0.08,
                    "retail_price": int(unit_cost * (1.7 + rng.random() * 0.45)),
                    "holding_cost_week": 5 + rng.integers(0, 9),
                    "lost_sale_penalty": 1200 if category_id == "CAT-DISH" else 900 if category_id == "CAT-SDA" else 650,
                    "case_pack": 6,
                    "units_per_container": 1800,
                }
            )
            sku_codes.append(sku)
    skus = pd.DataFrame(skus)

    trade_bu_ids = ["BU-CENTER", "BU-REGIONS", "BU-ONLINE"]
    planning_params = []
    for sku in skus["sku"]:
        for bu_id in trade_bu_ids:
            safety_stock = 130 + rng.integers(0, 220)
            max_stock = safety_stock * 3 + rng.integers(20, 200)
            planning_params.append(
                {
                    "sku": sku,
                    "bu_id": bu_id,
                    "safety_stock": safety_stock,
                    "max_stock": max_stock,
                    "is_critical": bool(rng.integers(0, 2)),
                    "granularity": "sku",
                    "params_updated_at": as_of.isoformat(),
                }
            )
    planning_params = pd.DataFrame(planning_params)

    sales_history = []
    for week_idx in range(52):
        week_start = as_of - timedelta(weeks=51 - week_idx)
        for _, row in skus.iterrows():
            season_weight = 1.0
            if row["category_id"] == "CAT-DECOR":
                season_weight = 1.6 if week_start.month in (11, 12) else 0.7 if week_start.month in (1, 2) else 1.0
            for bu_id in trade_bu_ids:
                base = 18 + (week_idx % 8) * 2 + (int(row["sku"].split("-")[-1]) % 10)
                promo_boost = 1.0
                if bu_id == "BU-ONLINE" and week_idx % 12 in (2, 5, 8):
                    promo_boost = 2.8
                actual_sales = max(0, int(base * season_weight * promo_boost * (0.8 + rng.random() * 0.55)))
                potential_demand = max(actual_sales, int(actual_sales * (1.08 + rng.random() * 0.35)))
                in_stock_share = round(float(np.clip(0.78 + rng.random() * 0.2, 0.6, 0.99)), 4)
                promo_id = None if week_idx % 12 not in (2, 5, 8) else f"PR-{bu_id[:2].lower()}-{week_idx}"
                sales_history.append(
                    {
                        "week_start": week_start.isoformat(),
                        "sku": row["sku"],
                        "bu_id": bu_id,
                        "actual_sales": actual_sales,
                        "potential_demand": potential_demand,
                        "in_stock_share": in_stock_share,
                        "promo_id": promo_id,
                    }
                )
    sales_history = pd.DataFrame(sales_history)

    inventory = []
    for _, row in skus.iterrows():
        for bu_id in trade_bu_ids:
            locations = [
                ("dc", "DC-MSK" if bu_id in {"BU-CENTER", "BU-ONLINE"} else "DC-EKB"),
                ("stores", f"STORES-{bu_id}"),
            ]
            if bu_id == "BU-ONLINE":
                locations.append(("marketplace", "MP-OZON-FBO"))
            for location_type, location_id in locations:
                on_hand = int(120 + rng.integers(30, 280) + (int(row["sku"].split("-")[-1]) % 20))
                reserved = int(rng.integers(0, 35))
                inventory.append(
                    {
                        "sku": row["sku"],
                        "bu_id": bu_id,
                        "location_type": location_type,
                        "location_id": location_id,
                        "on_hand": on_hand,
                        "reserved": reserved,
                        "transferable": location_type != "marketplace",
                        "updated_at": datetime.combine(as_of, datetime.min.time()).isoformat(timespec="seconds"),
                    }
                )
    inventory = pd.DataFrame(inventory)

    near_forecast = []
    for week_offset in range(1, 9):
        forecast_week = as_of + timedelta(weeks=week_offset)
        for _, row in skus.iterrows():
            for bu_id in trade_bu_ids:
                near_forecast.append(
                    {
                        "forecast_week": forecast_week.isoformat(),
                        "sku": row["sku"],
                        "bu_id": bu_id,
                        "forecast_qty": int(max(15, row["retail_price"] // 25) + rng.integers(0, 90)),
                        "created_week": as_of.isoformat(),
                    }
                )
    near_forecast = pd.DataFrame(near_forecast)

    sop_cycles = pd.DataFrame([{"cycle_id": "SOP-2026-10", "approved_at": as_of.isoformat(), "firm_lock_at": (as_of + timedelta(days=2)).isoformat(), "expedite_budget_rub": 1500000}])

    sop_weekly_plan = []
    for week_offset in range(12):
        week_start = as_of + timedelta(weeks=week_offset)
        for _, row in skus.iterrows():
            for bu_id in trade_bu_ids:
                qty = int(max(25, row["retail_price"] // 20) + rng.integers(25, 160))
                sop_weekly_plan.append(
                    {"cycle_id": "SOP-2026-10", "week_start": week_start.isoformat(), "sku": row["sku"], "bu_id": bu_id, "plan_qty": qty, "agreed_demand_qty": qty}
                )
    sop_weekly_plan = pd.DataFrame(sop_weekly_plan)

    firm_requests = []
    for sku in skus["sku"].head(10):
        for bu_id in trade_bu_ids:
            firm_requests.append(
                {
                    "cycle_id": "SOP-2026-10",
                    "sku": sku,
                    "bu_id": bu_id,
                    "need_week": (as_of + timedelta(weeks=3)).isoformat(),
                    "qty": 120 + rng.integers(0, 180),
                    "locked": True,
                }
            )
    firm_requests = pd.DataFrame(firm_requests)

    promo_registry = pd.DataFrame(
        [
            {"promo_id": "PR-ONL-11", "bu_id": "BU-ONLINE", "sku_list": ["SKU-DISH-001", "SKU-SDA-001"], "start_week": (as_of + timedelta(weeks=2)).isoformat(), "end_week": (as_of + timedelta(weeks=6)).isoformat(), "planned_uplift": 1.8, "commitment_type": "marketplace", "is_external_commitment": True, "approved_in_cycle": "SOP-2026-10", "min_shift_weeks": 2},
            {"promo_id": "PR-REG-07", "bu_id": "BU-REGIONS", "sku_list": ["SKU-DEC-001"], "start_week": (as_of + timedelta(weeks=5)).isoformat(), "end_week": (as_of + timedelta(weeks=8)).isoformat(), "planned_uplift": 1.6, "commitment_type": "leaflet", "is_external_commitment": True, "approved_in_cycle": "SOP-2026-10", "min_shift_weeks": 4},
            {"promo_id": "PR-CTR-03", "bu_id": "BU-CENTER", "sku_list": ["SKU-DISH-003"], "start_week": (as_of + timedelta(weeks=3)).isoformat(), "end_week": (as_of + timedelta(weeks=4)).isoformat(), "planned_uplift": 1.4, "commitment_type": "internal", "is_external_commitment": False, "approved_in_cycle": "SOP-2026-10", "min_shift_weeks": 0},
        ]
    )

    purchase_orders = []
    supplier_ids = suppliers["supplier_id"].tolist()
    for idx in range(24):
        supplier_id = supplier_ids[idx % len(supplier_ids)]
        sku = skus["sku"].iloc[idx % len(skus)]
        purchase_orders.append(
            {
                "po_id": f"PO-{idx + 1:04d}",
                "supplier_id": supplier_id,
                "sku": sku,
                "order_week": (as_of - timedelta(weeks=4 + (idx % 6))).isoformat(),
                "qty_total": 600 + idx * 25,
                "qty_base": 520 + idx * 20,
                "qty_extra": 80 + (idx % 10) * 5,
                "ready_week_planned": (as_of + timedelta(weeks=2 + (idx % 8))).isoformat(),
                "stage": ["in_production", "ready_not_shipped", "shipped"][idx % 3],
            }
        )
    purchase_orders = pd.DataFrame(purchase_orders)

    shipments = []
    for idx in range(20):
        shipments.append(
            {
                "shipment_id": f"CNT-{idx + 1:04d}",
                "po_ids": [f"PO-{(idx % 24) + 1:04d}"],
                "stage": ["ready_not_shipped", "sea", "rail", "port", "customs", "received_dc"][idx % 6],
                "mode": ["sea", "rail", "air"][idx % 3],
                "dest_dc_id": "DC-MSK" if idx % 2 == 0 else "DC-EKB",
                "ship_week": (as_of + timedelta(weeks=idx % 8)).isoformat(),
                "eta_at_ship": (as_of + timedelta(weeks=6 + (idx % 5))).isoformat(),
                "eta_current": (as_of + timedelta(weeks=7 + (idx % 6))).isoformat(),
                "eta_early": (as_of + timedelta(weeks=5 + (idx % 4))).isoformat(),
                "eta_base": (as_of + timedelta(weeks=6 + (idx % 5))).isoformat(),
                "eta_late": (as_of + timedelta(weeks=8 + (idx % 5))).isoformat(),
                "customs_hold_days": 3 + idx % 7,
                "shortfall_qty": 0,
                "defect_qty": 0,
                "expedite_options": ["ship_priority", "mode_switch"],
            }
        )
    shipments = pd.DataFrame(shipments)

    shipment_lines = []
    for _, shipment in shipments.iterrows():
        for bu_id in trade_bu_ids:
            sku = skus["sku"].iloc[(int(shipment["shipment_id"].split("-")[-1]) + len(bu_id)) % len(skus)]
            qty = 150 + rng.integers(0, 300)
            shipment_lines.append(
                {
                    "shipment_id": shipment["shipment_id"],
                    "sku": sku,
                    "bu_id": bu_id,
                    "qty_planned_for_bu": qty,
                    "qty_current_for_bu": qty,
                }
            )
    shipment_lines = pd.DataFrame(shipment_lines)

    freeze_zones = pd.DataFrame(
        [
            {"zone": "Z1_DC", "stages": ["port", "customs", "received_dc"], "levers": ["reallocate_bu", "customs_priority"], "decision_level": "import"},
            {"zone": "Z2_TRANSIT", "stages": ["sea", "rail"], "levers": ["redirect_dc", "ship_priority"], "decision_level": "session"},
            {"zone": "Z_SHIP", "stages": ["ready_not_shipped"], "levers": ["mode_switch", "air_part"], "decision_level": "session"},
            {"zone": "Z3_FACTORY", "stages": ["in_production"], "levers": ["qty_change", "cancel_order"], "decision_level": "sop"},
        ]
    )

    event_types = ["corridor_low", "sales_deviation", "promo_deviation", "eta_shift", "shortfall_defect", "customs_hold"]
    events = []
    for idx in range(40):
        sku = skus["sku"].iloc[idx % len(skus)]
        bu_id = trade_bu_ids[idx % len(trade_bu_ids)]
        trigger = event_types[idx % len(event_types)]
        shipment_id = f"CNT-{(idx % 20) + 1:04d}"
        shipment = shipments.iloc[idx % len(shipments)]
        eta_shift_days = (datetime.fromisoformat(shipment["eta_current"]) - datetime.fromisoformat(shipment["eta_at_ship"])).days
        fast_track = trigger in {"customs_hold", "shortfall_defect"} or (trigger == "eta_shift" and eta_shift_days > 5)
        detected_at = datetime.combine(as_of, datetime.min.time()) + timedelta(days=idx)
        economic_factor = 30 + (idx * 2) % 70
        urgency_factor = 25 + (idx * 3) % 75
        criticality_factor = 40 + (idx * 5) % 60
        risk_score = round(0.4 * economic_factor + 0.35 * urgency_factor + 0.25 * criticality_factor)
        priority = next(label for threshold, label in PRIORITY_LABELS if risk_score >= threshold)
        events.append(
            {
                "event_id": f"EVT-{idx + 1:04d}",
                "detected_at": detected_at.isoformat(timespec="minutes"),
                "contour": "bu" if idx % 3 != 0 else "import",
                "bu_id": bu_id,
                "side": ["demand", "supply", "conflict"][idx % 3],
                "trigger": trigger,
                "sku": sku,
                "shipment_id": shipment_id,
                "eta_shift_days": eta_shift_days,
                "promo_id": None,
                "impact_week": (as_of + timedelta(weeks=2 + idx % 6)).isoformat(),
                "weeks_to_impact": 1 + idx % 5,
                "projected_stock_min": 50 + idx * 3,
                "risk_rub": 150000 + idx * 12000,
                "economic_factor": economic_factor,
                "urgency_factor": urgency_factor,
                "criticality_factor": criticality_factor,
                "risk_score": risk_score,
                "priority": priority,
                "fast_track": fast_track,
                "status": ["new", "assessed", "in_contour", "to_session", "decided", "executed"][idx % 6],
                "owner_role": "BU_DM",
                "decision_deadline": (detected_at + timedelta(hours=48)).isoformat(timespec="minutes") if fast_track else (as_of + timedelta(days=4 + (idx % 5))).isoformat(),
                "root_cause": ["forecast_error", "promo_overperform", "supplier_delay", "customs", "marketplace_action"][idx % 5],
                "recurrence_weeks": 1 + idx % 4,
            }
        )
    events = pd.DataFrame(events)

    allocation_runs = pd.DataFrame(
        [{
            "run_id": "AL-001",
            "event_id": "EVT-0001",
            "sku": skus["sku"].iloc[0],
            "shipment_id": "CNT-0001",
            "available_qty": 1000,
            "basis": "agreed_demand",
            "step1_external_promo": 80,
            "step2_pro_rata": 460,
            "step_sell_through": 0,
            "result_by_bu": {"BU-CENTER": 400, "BU-ONLINE": 120, "BU-REGIONS": 180},
        }]
    )

    extra_needs = pd.DataFrame([
        {"need_id": "XN-0001", "bu_id": "BU-REGIONS", "sku": skus["sku"].iloc[0], "qty": 120, "need_week": (as_of + timedelta(weeks=2)).isoformat(), "reason": "Дополнительный спрос из-за промо", "submitted_at": (datetime.combine(as_of, datetime.min.time()) + timedelta(days=1)).isoformat(timespec="seconds"), "status": "open", "served_qty": 0, "source": "import_extra", "age_weeks": 1},
        {"need_id": "XN-0002", "bu_id": "BU-ONLINE", "sku": skus["sku"].iloc[1], "qty": 180, "need_week": (as_of + timedelta(weeks=3)).isoformat(), "reason": "Повышенный спрос маркетплейса", "submitted_at": (datetime.combine(as_of, datetime.min.time()) + timedelta(days=2)).isoformat(timespec="seconds"), "status": "served", "served_qty": 180, "source": "released_by_bu", "age_weeks": 1},
    ])

    request_releases = pd.DataFrame([
        {"release_id": "REL-001", "bu_id": "BU-CENTER", "sku": skus["sku"].iloc[0], "qty": 40, "released_at": (datetime.combine(as_of, datetime.min.time()) + timedelta(days=2)).isoformat(timespec="seconds"), "taken_by_bu_id": "BU-ONLINE", "compensation_rub": 4200},
    ])

    session_agenda = pd.DataFrame([
        {"session_week": as_of.isoformat(), "event_id": "EVT-0001", "order": 1, "prepared_options": ["opt-1"], "outcome": "decided"},
        {"session_week": as_of.isoformat(), "event_id": "EVT-0002", "order": 2, "prepared_options": ["opt-2"], "outcome": "escalated_owner"},
    ])

    kpi_snapshots = pd.DataFrame([
        {"kpi_id": "K01", "period_start": as_of.isoformat(), "scope": "group", "value": 0.92, "target": 0.94, "baseline": 0.88, "is_control_group": False},
        {"kpi_id": "K06", "period_start": as_of.isoformat(), "scope": "bu_id", "value": 0.11, "target": 0.1, "baseline": 0.12, "is_control_group": False},
    ])

    return {
        "business_units": business_units,
        "distribution_centers": distribution_centers,
        "categories": categories,
        "suppliers": suppliers,
        "skus": skus,
        "planning_params": planning_params,
        "sales_history": sales_history,
        "inventory": inventory,
        "near_forecast": near_forecast,
        "sop_cycles": sop_cycles,
        "sop_weekly_plan": sop_weekly_plan,
        "firm_requests": firm_requests,
        "promo_registry": promo_registry,
        "purchase_orders": purchase_orders,
        "shipments": shipments,
        "shipment_lines": shipment_lines,
        "freeze_zones": freeze_zones,
        "events": events,
        "allocation_runs": allocation_runs,
        "extra_needs": extra_needs,
        "request_releases": request_releases,
        "session_agenda": session_agenda,
        "kpi_snapshots": kpi_snapshots,
    }


def generate_data(as_of: date | None = None, *, mode: str = "legacy", domovoy: bool = False, profile: str | None = None, tz_v2: bool = False) -> dict[str, pd.DataFrame]:
    selected_mode = (profile or mode or "legacy").lower()
    if domovoy or tz_v2 or selected_mode in {"domovoy", "v2", "tz_v2", "demo_v2", "updated"}:
        return generate_domovoy_data(as_of)
    return generate_domovoy_data(as_of)


def _avg_last_demand(data: dict[str, pd.DataFrame], sku: str, bu_id: str) -> float:
    recent = data["sales_history"][(data["sales_history"]["sku"] == sku) & (data["sales_history"]["bu_id"] == bu_id)].tail(4)
    if recent.empty:
        return 0.0
    return float(recent["actual_sales"].mean())


def scenario_options(data: dict[str, pd.DataFrame], event: pd.Series, import_day: int) -> pd.DataFrame:
    """Return a small, explainable set of allocation options for the chosen event."""
    sku = event["sku"]
    bu_id = event.get("bu_id", "BU-CENTER")

    item = data["skus"][data["skus"]["sku"] == sku].iloc[0]
    inv = data["inventory"][(data["inventory"]["sku"] == sku) & (data["inventory"]["bu_id"] == bu_id)]
    if inv.empty:
        inv = data["inventory"][data["inventory"]["sku"] == sku].head(1)
    source_row = data["inventory"][(data["inventory"]["sku"] == sku) & (data["inventory"]["bu_id"] != bu_id)].head(1)
    source_free = 0
    if not source_row.empty:
        source_free = max(0, int(source_row.iloc[0]["on_hand"]) - int(source_row.iloc[0]["reserved"]) - int(source_row.iloc[0].get("safety_stock", 0)) if "safety_stock" in source_row.iloc[0] else 0)

    initial = max(0, int(inv.iloc[0]["on_hand"]) - int(inv.iloc[0]["reserved"]))
    demand = _avg_last_demand(data, sku, bu_id)
    if demand <= 0:
        demand = float(item["retail_price"]) / 25.0

    donor_qty = min(530, source_free)
    alternatives = [
        {"Вариант": "Без вмешательства", "Операция, ед.": 0, "Логистика, ₽": 0, "Закупка, ₽": 0, "Ограничение": "Базовый сценарий"},
        {"Вариант": "Межрегиональный трансфер", "Операция, ед.": donor_qty, "Логистика, ₽": donor_qty * 15, "Закупка, ₽": 0, "Ограничение": "Проверен запас источника; максимум 530 ед." if donor_qty else "Нет свободного ресурса после резервирования"},
        {"Вариант": "Срочное локальное пополнение", "Операция, ед.": 300, "Логистика, ₽": 0, "Закупка, ₽": 300 * int(item["unit_cost"]), "Ограничение": "Лимит срочного пополнения: 300 ед."},
    ]

    results = []
    for option in alternatives:
        qty = int(option["Операция, ед."])
        stock = initial
        deficit = 0.0
        deficit_before = 0.0
        min_stock = stock
        stock_days = 0
        for day in range(1, 29):
            if day == import_day:
                stock += qty
            if day >= import_day:
                fulfilled = min(stock, demand)
                deficit += max(0.0, demand - fulfilled)
                stock = max(0, stock - fulfilled)
            else:
                fulfilled = min(stock, demand)
                deficit_before += max(0.0, demand - fulfilled)
                stock = max(0, stock - fulfilled)
            min_stock = min(min_stock, stock)
            stock_days += stock
        total_cost = int(
            deficit * float(item["lost_sale_penalty"])
            + stock_days * float(item["holding_cost_week"]) / 28
            + option["Логистика, ₽"]
            + option["Закупка, ₽"]
        )
        service = max(0.0, 100 * (1 - deficit / max(demand * 28, 1)))
        service_before = max(0.0, 100 * (1 - deficit_before / max(demand * max(0, import_day - 1), 1)))
        results.append(
            {
                "Вариант": option["Вариант"],
                "Выполнение спроса, %": round(service, 1),
                "Сервис до импорта, %": round(service_before, 1),
                "Дефицит, ед.": int(round(deficit)),
                "Дефицит до импорта, ед.": int(round(deficit_before)),
                "Остаток на конец, ед.": int(stock),
                "Минимальный остаток, ед.": int(min_stock),
                "Логистика, ₽": int(option["Логистика, ₽"]),
                "Закупка, ₽": int(option["Закупка, ₽"]),
                "Итого, ₽": int(total_cost),
                "Операция, ед.": qty,
                "Ограничение": option["Ограничение"],
            }
        )

    return pd.DataFrame(results)


def optimize_transfer(data: dict[str, pd.DataFrame], event: pd.Series, import_day: int) -> dict[str, Any]:
    """Choose the cheapest feasible option across the generated scenario table."""
    try:
        import pulp
    except ImportError:
        return {"status": "unavailable", "message": "PuLP не установлен. Установите зависимости из requirements.txt и повторите расчёт."}

    options = scenario_options(data, event, import_day)
    if options.empty:
        return {"status": "failed", "message": "Нет доступных вариантов для расчёта."}

    model = pulp.LpProblem("soe_domovoy_choice", pulp.LpMinimize)
    choices = [pulp.LpVariable(f"alt_{idx}", cat="Binary") for idx in range(len(options))]
    model += pulp.lpSum(float(cost) * choice for cost, choice in zip(options["Итого, ₽"].tolist(), choices))
    model += pulp.lpSum(choices) == 1

    for idx, row in options.iterrows():
        if row["Вариант"] == "Межрегиональный трансфер":
            model += int(row["Операция, ед."]) <= 530

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return {"status": "failed", "message": f"Оптимизатор не нашёл оптимальное решение: {pulp.LpStatus[status]}"}

    selected_idx = next(i for i, choice in enumerate(choices) if pulp.value(choice) == 1)
    return {"status": "optimal", "message": "Решение найдено CBC для демонстрационной модели.", "option": options.iloc[selected_idx].to_dict()}
