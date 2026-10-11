from __future__ import annotations

import pandas as pd
import streamlit as st

from .model import generate_domovoy_data, optimize_transfer, scenario_options


st.set_page_config(page_title="Домовой S&OE", layout="wide")

TRIGGER_LABELS = {
    "corridor_low": "Риск дефицита",
    "sales_deviation": "Отклонение продаж",
    "promo_deviation": "Отклонение промо",
    "eta_shift": "Сдвиг ETA",
    "shortfall_defect": "Недопоставка / дефект",
    "customs_hold": "Задержка на таможне",
}
STATUS_LABELS = {
    "new": "Новое",
    "assessed": "Оценено",
    "in_contour": "В работе БЕ",
    "to_session": "На совещание",
    "decided": "Решено",
    "executed": "Исполнено",
}
STAGE_LABELS = {
    "in_production": "Производство",
    "ready_not_shipped": "Готово к отгрузке",
    "sea": "Море",
    "rail": "Ж/д",
    "port": "Порт",
    "customs": "Таможня",
    "received_dc": "Центральный РЦ",
}
LEVER_LABELS = {
    "reallocate_bu": "Перераспределение между БЕ",
    "customs_priority": "Приоритет растаможки",
    "redirect_dc": "Перенаправление на другой РЦ",
    "ship_priority": "Приоритет отгрузки",
    "mode_switch": "Смена вида транспорта",
    "air_part": "Авиадоставка части партии",
    "qty_change": "Изменение объёма заказа",
    "cancel_order": "Отмена заказа",
}
DECISION_LEVEL_LABELS = {"import": "БЕ «Импорт»", "session": "Совместное совещание", "sop": "S&OP"}


def _render_filters(data: dict[str, pd.DataFrame]) -> tuple[str, str, int]:
    business_units = data["business_units"]
    categories = data["categories"]
    bu_names = dict(zip(business_units["bu_id"], business_units["name"]))
    category_names = dict(zip(categories["category_id"], categories["name"]))

    st.sidebar.header("Срез данных")
    selected_bu = st.sidebar.selectbox(
        "Бизнес-единица",
        ["Все", *business_units["bu_id"].tolist()],
        format_func=lambda value: "Все БЕ" if value == "Все" else bu_names[value],
    )
    selected_category = st.sidebar.selectbox(
        "Категория",
        ["Все", *categories["category_id"].tolist()],
        format_func=lambda value: "Все категории" if value == "Все" else category_names[value],
    )
    history_weeks = st.sidebar.select_slider("Период истории", options=[13, 26, 52], value=26)
    st.sidebar.caption("Синтетический демонстрационный контур")
    return selected_bu, selected_category, history_weeks


def _apply_scope(frame: pd.DataFrame, selected_bu: str, selected_category: str) -> pd.DataFrame:
    result = frame
    if selected_bu != "Все" and "bu_id" in result.columns:
        result = result[result["bu_id"] == selected_bu]
    if selected_category != "Все" and "category_id" in result.columns:
        result = result[result["category_id"] == selected_category]
    return result


def _prepare_events(data: dict[str, pd.DataFrame], selected_bu: str, selected_category: str) -> pd.DataFrame:
    events = data["events"].merge(data["skus"][["sku", "category_id"]], on="sku")
    shipment_details = data["shipments"][
        ["shipment_id", "stage", "dest_dc_id", "eta_current", "eta_at_ship"]
    ]
    events = events.merge(shipment_details, on="shipment_id", how="left")
    return _apply_scope(events, selected_bu, selected_category)


def _render_problem_panel(events: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    st.title("Домовой | Панель проблем")
    st.caption("Операционный S&OE-контур: выявить риск, назначить владельца, сравнить допустимые меры")
    st.info("Объём закупки не меняется в S&OE. После отгрузки доступны только действия по сроку, маршруту и распределению.")

    cadence = st.columns(5)
    for column, label in zip(cadence, ["Пн · данные", "Вт · регионы", "Ср · импорт", "Чт · совещание", "Пт · исполнение"]):
        column.markdown(f"**{label}**")
    st.caption("Четверговая встреча: до 60 минут, в повестке не более 8 неразрешённых споров.")

    active_events = events[~events["status"].isin(["decided", "executed"])].copy()
    fast_track_events = active_events[active_events["fast_track"]]
    session_events = active_events[active_events["status"] == "to_session"]
    metrics = st.columns(4)
    metrics[0].metric("Проблемы в работе", len(active_events))
    metrics[1].metric("Деньги под риском", f"{active_events['risk_rub'].sum() / 1_000_000:.1f} млн ₽")
    metrics[2].metric("Экспресс-трек", len(fast_track_events), help="Порт/дефект или сдвиг ETA более 5 дней; срок решения — 48 часов.")
    metrics[3].metric("На четверговой встрече", len(session_events), delta="лимит 8 случаев")

    if active_events.empty:
        st.success("По выбранному срезу открытых проблем нет.")
        return

    priority_queue = active_events.sort_values(["risk_rub", "risk_score"], ascending=False).head(8).copy()
    st.subheader("Приоритетная очередь")
    st.caption("Кандидаты для разбора отсортированы по сумме риска; окончательное решение остаётся за владельцами процесса.")
    queue_display = priority_queue[["event_id", "trigger", "bu_id", "sku", "priority", "risk_rub", "stage", "status", "fast_track"]].copy()
    queue_display["trigger"] = queue_display["trigger"].map(TRIGGER_LABELS)
    queue_display["stage"] = queue_display["stage"].map(STAGE_LABELS)
    queue_display["status"] = queue_display["status"].map(STATUS_LABELS)
    queue_display["risk_rub"] = queue_display["risk_rub"].map(lambda value: f"{value:,.0f} ₽".replace(",", " "))
    queue_display["fast_track"] = queue_display["fast_track"].map({True: "≤ 48 ч", False: "Обычный"})
    queue_display = queue_display.rename(columns={
        "event_id": "Кейс", "trigger": "Проблема", "bu_id": "БЕ", "sku": "SKU",
        "priority": "Приоритет", "risk_rub": "Риск", "stage": "Этап поставки",
        "status": "Статус", "fast_track": "Срок",
    })
    st.dataframe(queue_display, width="stretch", hide_index=True)

    event_labels = {
        row["event_id"]: f"{row['event_id']} · {TRIGGER_LABELS[row['trigger']]} · {row['risk_rub'] / 1000:.0f} тыс. ₽"
        for _, row in priority_queue.iterrows()
    }
    selected_event_id = st.selectbox(
        "Открыть кейс",
        priority_queue["event_id"].tolist(),
        format_func=lambda value: event_labels[value],
    )
    event = priority_queue[priority_queue["event_id"] == selected_event_id].iloc[0]
    shipment_stage = event["stage"]
    zone = next(
        (row for _, row in data["freeze_zones"].iterrows() if shipment_stage in row["stages"]),
        None,
    )

    st.subheader(f"Карточка кейса {event['event_id']}")
    case_metrics = st.columns(5)
    case_metrics[0].metric("Причина", TRIGGER_LABELS[event["trigger"]])
    case_metrics[1].metric("Владелец", event["owner_role"])
    case_metrics[2].metric("Влияние через", f"{event['weeks_to_impact']} нед.")
    case_metrics[3].metric("Сдвиг ETA", f"{event['eta_shift_days']} дн.")
    case_metrics[4].metric("Текущий этап", STAGE_LABELS[shipment_stage])

    if zone is not None:
        permitted_actions = ", ".join(LEVER_LABELS[lever] for lever in zone["levers"])
        st.info(
            f"Зона {zone['zone']} · решение: {DECISION_LEVEL_LABELS[zone['decision_level']]}. "
            f"Доступные меры: {permitted_actions}."
        )
        if zone["decision_level"] == "sop":
            st.warning("Изменение объёма или отмена заказа эскалируются в месячный S&OP.")

    deadline = pd.to_datetime(event["decision_deadline"]).strftime("%d.%m.%Y %H:%M")
    st.caption(f"Срок решения: {deadline} · статус: {STATUS_LABELS[event['status']]} · неделя влияния: {event['impact_week']}")

    import_day = st.slider("День поступления в сценарии", min_value=3, max_value=21, value=10)
    options = scenario_options(data, event, import_day)
    st.dataframe(options, width="stretch", hide_index=True)

    result = optimize_transfer(data, event, import_day)
    if result["status"] == "optimal":
        option = result["option"]
        st.success(result["message"])
        result_cols = st.columns(4)
        result_cols[0].metric("Рекомендуемый вариант", option["Вариант"])
        result_cols[1].metric("Полные затраты", f"{option['Итого, ₽']:,.0f} ₽".replace(",", " "))
        result_cols[2].metric("Выполнение спроса", f"{option['Выполнение спроса, %']:.1f}%")
        result_cols[3].metric("Дефицит", f"{option['Дефицит, ед.']} ед.")
    elif result["status"] == "unavailable":
        st.warning(result["message"])
    else:
        st.error(result["message"])


def _render_flow(data: dict[str, pd.DataFrame]) -> None:
    st.title("Поток поставки и правила решений")
    st.caption("Рычаги S&OE ограничены текущим этапом партии; объём закупки остаётся в контуре S&OP.")

    stage_order = ["ready_not_shipped", "sea", "rail", "port", "customs", "received_dc"]
    shipment_counts = data["shipments"].groupby("stage").size().reindex(stage_order, fill_value=0)
    pipeline = pd.DataFrame({
        "Этап": [STAGE_LABELS[stage] for stage in stage_order],
        "Партий": shipment_counts.tolist(),
    })
    st.bar_chart(pipeline, x="Этап", y="Партий", height=300)

    zones = data["freeze_zones"].copy()
    zones["Этапы"] = zones["stages"].map(lambda values: ", ".join(STAGE_LABELS[value] for value in values))
    zones["Допустимые меры"] = zones["levers"].map(lambda values: ", ".join(LEVER_LABELS[value] for value in values))
    zones["Кто решает"] = zones["decision_level"].map(DECISION_LEVEL_LABELS)
    st.dataframe(zones[["zone", "Этапы", "Допустимые меры", "Кто решает"]].rename(columns={"zone": "Зона заморозки"}), width="stretch", hide_index=True)

    st.subheader("Недельный ритм")
    cadence = pd.DataFrame([
        {"День": "Понедельник", "Работа": "Обновление данных и панели проблем", "Владелец": "Команда данных"},
        {"День": "Вторник", "Работа": "Разбор дефицитов и заявок регионов", "Владелец": "Торговые БЕ"},
        {"День": "Среда", "Работа": "Сроки прибытия и меры по импорту", "Владелец": "БЕ «Импорт»"},
        {"День": "Четверг", "Работа": "Только неразрешённые споры, до 8 случаев", "Владелец": "Нейтральный координатор"},
        {"День": "Пятница", "Работа": "Исполнение решений и контроль", "Владелец": "Владельцы решений"},
    ])
    st.dataframe(cadence, width="stretch", hide_index=True)


def _render_analytics(data: dict[str, pd.DataFrame], selected_bu: str, selected_category: str, history_weeks: int) -> None:
    st.title("Аналитика спроса и запасов")
    categories = data["categories"]

    sales = data["sales_history"].merge(data["skus"][["sku", "category_id"]], on="sku")
    sales = _apply_scope(sales, selected_bu, selected_category)
    sales["week_start"] = pd.to_datetime(sales["week_start"])
    if not sales.empty:
        cutoff = sales["week_start"].max() - pd.Timedelta(weeks=history_weeks - 1)
        sales = sales[sales["week_start"] >= cutoff]
    sales_trend = (
        sales.groupby("week_start", as_index=False)[["actual_sales", "potential_demand"]]
        .sum()
        .rename(columns={"actual_sales": "Факт продаж", "potential_demand": "Потенциальный спрос"})
    )

    stock = data["inventory"].drop_duplicates(["sku", "bu_id", "location_id"]).copy()
    stock["available_qty"] = (stock["on_hand"] - stock["reserved"]).clip(lower=0)
    stock = stock.groupby(["sku", "bu_id"], as_index=False)["available_qty"].sum()
    stock = stock.merge(data["planning_params"][["sku", "bu_id", "safety_stock"]], on=["sku", "bu_id"])
    stock = stock.merge(data["skus"][["sku", "category_id"]], on="sku")
    stock = _apply_scope(stock, selected_bu, selected_category)
    stock_by_category = (
        stock.merge(categories[["category_id", "name"]], on="category_id")
        .groupby(["category_id", "name"], as_index=False)[["available_qty", "safety_stock"]]
        .sum()
        .rename(columns={"name": "Категория", "available_qty": "Доступный остаток", "safety_stock": "Страховой запас"})
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Продажи и потенциальный спрос")
        st.line_chart(sales_trend, x="week_start", y=["Факт продаж", "Потенциальный спрос"], height=320)
    with right:
        st.subheader("Остаток против страхового запаса")
        st.bar_chart(stock_by_category, x="Категория", y=["Доступный остаток", "Страховой запас"], height=320)

    events = _prepare_events(data, selected_bu, selected_category)
    active_events = events[~events["status"].isin(["decided", "executed"])]
    priority_order = ["Критический", "Высокий", "Средний", "Низкий"]
    priority_counts = (
        active_events.groupby("priority").size().reindex(priority_order, fill_value=0)
        .rename_axis("Приоритет").reset_index(name="Событий")
    )
    st.subheader("Открытые проблемы по приоритету")
    st.bar_chart(priority_counts, x="Приоритет", y="Событий", height=280)
    st.caption("Доступность в графиках — среднее значение синтетического in-stock share, не целевой KPI пилота.")


def _render_pilot_targets() -> None:
    st.title("Цели и критерии пилота")
    st.caption("Ориентиры из питч-дека. Это целевые значения проекта, а не факт синтетической модели.")
    st.metric("Ожидаемый годовой эффект", "≈ 1,3% выручки импорта")

    targets = pd.DataFrame([
        {"Показатель": "Наличие импорта в магазине", "Было": "88%", "Цель": "91%", "Направление": "Рост"},
        {"Показатель": "Непроданный остаток в конце сезона", "Было": "25%", "Цель": "20%", "Направление": "Снижение"},
        {"Показатель": "Контейнеры со срочными мерами", "Было": "12,5%", "Цель": "9%", "Направление": "Снижение"},
        {"Показатель": "Запас импорта", "Было": "120 дней", "Цель": "105 дней", "Направление": "Снижение"},
        {"Показатель": "Отклонение заявок регионов от продаж", "Было": "+18%", "Цель": "±10%", "Направление": "Сближение"},
        {"Показатель": "Время от сбоя до решения", "Было": "2–3 недели", "Цель": "≤ 48 ч", "Направление": "Сокращение"},
    ])
    st.dataframe(targets, width="stretch", hide_index=True)

    st.subheader("Контрольные результаты внедрения")
    outcomes = st.columns(4)
    outcomes[0].metric("Импорт в цикле к 14-му месяцу", "100%")
    outcomes[1].metric("Проблемы решены за неделю", "≥ 85%")
    outcomes[2].metric("Экспресс-решение", "≤ 48 ч")
    outcomes[3].metric("Решается автоматически", "≥ 15%")

    st.subheader("Этапы и контрольные точки · 18 месяцев")
    roadmap = pd.DataFrame([
        {"Этап": "Диагностика", "Срок": "1–3 мес.", "Результат / gate": "Базовые KPI и правила утверждены S&OP"},
        {"Этап": "Пилот", "Срок": "4–9 мес.", "Результат / gate": "Две категории; наличие выше контроля на 1,5 п.п."},
        {"Этап": "Тиражирование", "Срок": "10–14 мес.", "Результат / gate": "Волна 1 — 50%, затем весь импорт в цикле"},
        {"Этап": "Автоматизация", "Срок": "13–18 мес.", "Результат / gate": "Автоправила и сценарии после проверки пилота"},
    ])
    st.dataframe(roadmap, width="stretch", hide_index=True)
    st.subheader("Приёмка")
    st.markdown(
        "- Импорт участвует в цикле не менее 3 месяцев.\n"
        "- Выполнено не менее 4 из 6 целей, причём цель по наличию обязательна.\n"
        "- Регламент, RACI и правила утверждены S&OP; панель работает штатно.\n"
        "- Зрелость S&OE достигла уровня 4 из 5.\n"
        "- Эффект измеряется сравнением с контрольными категориями и предыдущим сезоном."
    )


def run() -> None:
    data = generate_domovoy_data()
    selected_bu, selected_category, history_weeks = _render_filters(data)
    events = _prepare_events(data, selected_bu, selected_category)

    problems_tab, flow_tab, analytics_tab, targets_tab, data_tab = st.tabs(
        ["Панель проблем", "Поток и правила", "Аналитика", "Цели пилота", "Данные"]
    )
    with problems_tab:
        _render_problem_panel(events, data)
    with flow_tab:
        _render_flow(data)
    with analytics_tab:
        _render_analytics(data, selected_bu, selected_category, history_weeks)
    with targets_tab:
        _render_pilot_targets()
    with data_tab:
        st.subheader("История продаж")
        st.dataframe(data["sales_history"].head(20), width="stretch", hide_index=True)
        st.subheader("Запасы")
        st.dataframe(data["inventory"].head(20), width="stretch", hide_index=True)


if __name__ == "__main__":
    run()
