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
    assert {"Без вмешательства", "Межрегиональный трансфер", "Срочное локальное пополнение"}.issubset(set(options["Вариант"]))
    assert options["Итого, ₽"].min() >= 0
    assert options["Выполнение спроса, %"].between(0, 100).all()


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
