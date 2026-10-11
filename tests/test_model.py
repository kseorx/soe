from datetime import date

import pandas as pd

from src.model import (
    allocate_equal_cover_days,
    cap_extra_import_quantity,
    generate_domovoy_data,
    optimize_transfer,
    scenario_options,
)


def test_domovoy_data_shape():
    data = generate_domovoy_data(date(2026, 10, 5))

    assert len(data["business_units"]) == 6
    assert (data["business_units"]["bu_type"] == "trade").sum() == 5
    assert set(data["business_units"].loc[data["business_units"]["bu_type"] == "trade", "name"]) == {
        f"Региональный дивизион {index}" for index in range(1, 6)
    }
    assert "Импорт" in set(data["business_units"]["name"])
    assert not set(data["business_units"]["name"]) & {"Центр", "Регионы", "Онлайн"}
    assert set(data["promo_registry"]["bu_id"]).issubset(set(data["business_units"]["bu_id"]))
    factory_zone = data["freeze_zones"].set_index("zone").loc["Z3_FACTORY"]
    assert factory_zone["decision_level"] == "sop"
    assert set(factory_zone["levers"]) == {"qty_change", "cancel_order"}
    assert len(data["distribution_centers"]) == 2
    assert len(data["skus"]) == 30
    assert data["skus"]["category_id"].nunique() == 3
    assert len(data["sales_history"]) == 52 * 30 * 5
    assert not data["inventory"].duplicated(["sku", "bu_id", "location_id"]).any()
    expected_fast_track = (
        data["events"]["trigger"].isin(["customs_hold", "shortfall_defect"])
        | ((data["events"]["trigger"] == "eta_shift") & (data["events"]["eta_shift_days"] > 5))
    )
    assert data["events"]["fast_track"].equals(expected_fast_track)
    assert len(data["purchase_orders"]) == 24
    assert len(data["shipments"]) == 20
    assert len(data["events"]) >= 40


def test_scenario_options_are_feasible_and_ordered():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[0]

    options = scenario_options(data, event, import_day=10)

    assert not options.empty
    assert set(options["Вариант"]) == {
        "Без вмешательства",
        "Межрегиональный трансфер из свободного запаса",
    }
    assert "Закупка, ₽" not in options.columns
    assert options["Итого, ₽"].min() >= 0
    assert options["Выполнение спроса, %"].between(0, 100).all()


def test_scenario_uses_daily_demand_and_dc_stock_for_screenshot_case():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[7]

    options = scenario_options(data, event, import_day=5)
    baseline = options[options["Вариант"] == "Без вмешательства"].iloc[0]
    recent_sales = data["sales_history"]
    recent_sales = recent_sales[(recent_sales["sku"] == event["sku"]) & (recent_sales["bu_id"] == event["bu_id"])].tail(4)
    daily_demand = float(recent_sales["potential_demand"].mean()) / 7
    dc_stock = data["inventory"]
    dc_stock = dc_stock[(dc_stock["sku"] == event["sku"]) & (dc_stock["bu_id"] == event["bu_id"]) & (dc_stock["location_type"] == "dc")]
    available_stock = int((dc_stock["on_hand"] - dc_stock["reserved"]).clip(lower=0).sum())
    expected_deficit = max(0, round(daily_demand * 28 - available_stock))

    assert baseline["Дефицит, ед."] == expected_deficit
    assert baseline["Спрос за 28 дней, ед."] == round(daily_demand * 28)
    assert baseline["Запас РЦ на старте, ед."] == available_stock
    assert baseline["Выполнение спроса, %"] == round(100 * (1 - expected_deficit / (daily_demand * 28)), 1)
    assert baseline["Дефицит до импорта, ед."] <= baseline["Дефицит, ед."]
    feasible_options = options[
        (options["Вариант"] == "Без вмешательства") | (options["Операция, ед."] > 0)
    ]
    recommendation = optimize_transfer(data, event, import_day=5)
    assert recommendation["option"]["Итого, ₽"] == feasible_options["Итого, ₽"].min()


def test_screenshot_event_uses_potential_demand_and_reports_covered_stock():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[39]
    options = scenario_options(data, event, import_day=5)
    result = options.iloc[0]
    recent_sales = data["sales_history"]
    recent_sales = recent_sales[(recent_sales["sku"] == event["sku"]) & (recent_sales["bu_id"] == event["bu_id"])].tail(4)
    expected_demand = round(float(recent_sales["potential_demand"].mean()) * 4)

    assert result["Запас РЦ на старте, ед."] > 0
    assert result["Спрос за 28 дней, ед."] == expected_demand
    assert result["Дефицит, ед."] == max(0, expected_demand - result["Запас РЦ на старте, ед."])
    assert result["Выполнение спроса, %"] == round(100 * (1 - result["Дефицит, ед."] / expected_demand), 1)


def test_transfer_uses_only_donor_stock_above_safety_stock():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[0]
    target_dc = (data["inventory"]["sku"] == event["sku"]) & (data["inventory"]["bu_id"] == event["bu_id"]) & (data["inventory"]["location_type"] == "dc")
    data["inventory"].loc[target_dc, "on_hand"] = 0

    options = scenario_options(data, event, import_day=5)
    transfer = options[options["Вариант"] == "Межрегиональный трансфер из свободного запаса"].iloc[0]
    assert transfer["Операция, ед."] > 0
    assert transfer["Операция, ед."] <= 530
    recommendation = optimize_transfer(data, event, import_day=5)
    assert recommendation["option"]["Вариант"] == "Межрегиональный трансфер из свободного запаса"


def test_optimizer_returns_structured_result():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[0]

    result = optimize_transfer(data, event, import_day=10)

    assert result["status"] in {"optimal", "unavailable", "failed"}
    if result["status"] == "optimal":
        assert "option" in result
        assert "Итого, ₽" in result["option"]


def test_seed_reproducibility():
    one = generate_domovoy_data(date(2026, 10, 5))
    two = generate_domovoy_data(date(2026, 10, 5))

    pd.testing.assert_frame_equal(one["events"], two["events"])
    pd.testing.assert_frame_equal(one["inventory"], two["inventory"])


def test_equal_cover_allocator_levels_stock_days_within_request_caps():
    allocation = allocate_equal_cover_days(
        available_qty=100,
        current_stock={"DIV-A": 100, "DIV-B": 10, "DIV-C": 20},
        daily_demand={"DIV-A": 10, "DIV-B": 5, "DIV-C": 2},
        request_caps={"DIV-A": 100, "DIV-B": 100, "DIV-C": 100},
    )

    assert allocation["allocation_qty"].sum() == 100
    assert allocation["allocation_qty"].between(0, 100).all()
    assert allocation["coverage_days_after"].max() - allocation["coverage_days_after"].min() <= 0.2


def test_equal_cover_allocator_does_not_exceed_firm_requests():
    allocation = allocate_equal_cover_days(
        available_qty=500,
        current_stock={"DIV-A": 0, "DIV-B": 0},
        daily_demand={"DIV-A": 10, "DIV-B": 5},
        request_caps={"DIV-A": 50, "DIV-B": 70},
    )

    assert allocation["allocation_qty"].sum() == 120
    assert allocation.set_index("bu_id")["allocation_qty"].to_dict() == {"DIV-A": 50, "DIV-B": 70}


def test_extra_import_is_capped_at_ten_percent_of_confirmed_requests():
    assert cap_extra_import_quantity(1000, 500) == 100
    assert cap_extra_import_quantity(1000, 50) == 50
    assert cap_extra_import_quantity(0, 500) == 0
