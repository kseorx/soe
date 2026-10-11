from datetime import date

import pandas as pd

from src.model import generate_domovoy_data, optimize_transfer, scenario_options


def test_domovoy_data_shape():
    data = generate_domovoy_data(date(2026, 10, 5))

    assert len(data["business_units"]) >= 4
    assert len(data["distribution_centers"]) == 2
    assert len(data["skus"]) == 30
    assert data["skus"]["category_id"].nunique() == 3
    assert len(data["sales_history"]) == 52 * 30 * 3
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
    assert baseline["Вариант"] == optimize_transfer(data, event, import_day=5)["option"]["Вариант"]


def test_screenshot_event_uses_potential_demand_and_reports_covered_stock():
    data = generate_domovoy_data(date(2026, 10, 5))
    event = data["events"].iloc[39]
    result = scenario_options(data, event, import_day=5).iloc[0]

    assert result["Выполнение спроса, %"] == 100.0
    assert result["Запас РЦ на старте, ед."] == 352
    assert result["Спрос за 28 дней, ед."] == 117
    assert result["Дефицит, ед."] == 0


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
