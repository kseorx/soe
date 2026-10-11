from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .model import (
    allocate_equal_cover_days,
    cap_extra_import_quantity,
    generate_domovoy_data,
    optimize_transfer,
    scenario_options,
)


st.set_page_config(page_title="Домовой S&OE", layout="wide")

NAVY = "#263A82"
INDIGO = "#5969C6"
MID_BLUE = "#8794D3"
PALE_BLUE = "#E8ECF5"
AMBER = "#F5A000"
INK = "#24344E"
MUTED = "#68788E"
GRID = "#DCE2EB"


def _apply_visual_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #F2F3F6; color: #24344E; }
        [data-testid="stHeader"] { background: rgba(242, 243, 246, 0.92); }
        [data-testid="stSidebar"] { background: #E8ECF5; }
        h1, h2, h3 { color: #263A82; font-family: 'Aptos Display', 'Segoe UI', sans-serif; }
        h1 { font-weight: 400; }
        [data-testid="stMetric"] {
            background: #FFFFFF; border-left: 4px solid #5969C6;
            padding: 12px 16px; border-radius: 4px;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #263A82; border-bottom-color: #F5A000;
        }
        [data-testid="stDataFrame"] { border: 1px solid #DCE2EB; }
        .soe-flow-zones { display: grid; grid-template-columns: 1fr 4fr 1fr; gap: 8px; margin: 12px 0; }
        .soe-flow-zones span { padding: 8px; text-align: center; font-size: 13px; font-weight: 600; }
        .soe-flow-sop { background: #E8ECF5; color: #68788E; }
        .soe-flow-soe { background: #5969C6; color: #FFFFFF; }
        .soe-flow-sales { background: #E8ECF5; color: #68788E; }
        .soe-flow-steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
        .soe-flow-step { min-height: 88px; padding: 12px; background: #FFFFFF; border-top: 4px solid #5969C6; border-radius: 3px; }
        .soe-flow-step:nth-child(3), .soe-flow-step:nth-child(5) { border-top-color: #F5A000; }
        .soe-flow-step b { display: block; color: #263A82; font-size: 14px; line-height: 1.3; }
        .soe-flow-step small { display: block; margin-top: 6px; color: #68788E; font-size: 12px; }
        .soe-flow-step i { display: block; margin-bottom: 5px; color: #68788E; font-size: 11px; font-style: normal; }
        @media (max-width: 700px) {
            [data-testid="stAppViewBlockContainer"] { padding-left: 0.75rem; padding-right: 0.75rem; }
            .soe-flow-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
            .soe-flow-step { min-height: 82px; padding: 10px; }
            .soe-flow-zones span { font-size: 11px; padding: 7px 3px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _chart_layout(figure: go.Figure, *, height: int = 340) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 20, "r": 20, "t": 22, "b": 24},
        font={"family": "Aptos, Segoe UI, sans-serif", "color": INK, "size": 12},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#FFFFFF",
        colorway=[INDIGO, AMBER, MID_BLUE, NAVY],
        xaxis={"gridcolor": GRID, "zerolinecolor": GRID},
        yaxis={"gridcolor": GRID, "zerolinecolor": GRID},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return figure

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
    eta_shift_days = int(event["eta_shift_days"])
    eta_value = f"+{eta_shift_days} дн." if eta_shift_days > 0 else f"{abs(eta_shift_days)} дн."
    eta_delta = "позже плана" if eta_shift_days > 0 else "раньше плана" if eta_shift_days < 0 else "по плану"
    metric_rows = [st.columns(2) for _ in range(4)]
    metric_rows[0][0].metric("Причина", TRIGGER_LABELS[event["trigger"]])
    metric_rows[0][1].metric("Ответственный", event["owner_role"])
    metric_rows[1][0].metric("Бизнес-единица", event["bu_id"])
    metric_rows[1][1].metric("До влияния", f"{event['weeks_to_impact']} нед.")
    metric_rows[2][0].metric("Отклонение ETA", eta_value, delta=eta_delta)
    metric_rows[2][1].metric("Этап поставки", STAGE_LABELS[shipment_stage])
    metric_rows[3][0].metric("Риск события", f"{event['risk_rub'] / 1000:.0f} тыс. ₽")
    metric_rows[3][1].metric("Приоритет", event["priority"])

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
    st.subheader("Сравнение сценариев · горизонт 28 дней")
    st.caption("Продажи заданы за неделю и переведены в дневной спрос. Варианты новой закупки не рассматриваются: S&OE работает с уже купленным товаром.")
    visible_options = options[
        (options["Вариант"] == "Без вмешательства") | (options["Операция, ед."] > 0)
    ]
    service_chart = go.Figure(go.Bar(
        x=visible_options["Выполнение спроса, %"],
        y=visible_options["Вариант"],
        orientation="h",
        marker_color=[MUTED if name == "Без вмешательства" else INDIGO for name in visible_options["Вариант"]],
        text=[f"{value:.1f}%" for value in visible_options["Выполнение спроса, %"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% спроса выполнено<extra></extra>",
    ))
    service_chart.update_layout(
        xaxis={"title": "Выполнение спроса", "range": [0, 112], "ticksuffix": "%"},
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    st.plotly_chart(_chart_layout(service_chart, height=190 if len(visible_options) == 1 else 240), width="stretch", config={"displayModeBar": False})
    for _, scenario in visible_options.iterrows():
        with st.expander(f"{scenario['Вариант']} · {scenario['Выполнение спроса, %']:.1f}% спроса"):
            detail_cols = st.columns(2)
            detail_cols[0].write(f"Потенциальный спрос за 28 дней: **{scenario['Спрос за 28 дней, ед.']} ед.**")
            detail_cols[1].write(f"Доступно на РЦ: **{scenario['Запас РЦ на старте, ед.']} ед.**")
            detail_cols[0].write(f"Дефицит за 28 дней: **{scenario['Дефицит, ед.']} ед.**")
            detail_cols[1].write(f"Общие модельные затраты: **{scenario['Итого, ₽']:,.0f} ₽**".replace(",", " "))
            detail_cols[0].write(f"Операция: **{scenario['Операция, ед.']} ед.**")
            detail_cols[1].write(f"Трансфер: **{scenario['Логистика, ₽']:,.0f} ₽**".replace(",", " "))
            st.caption(scenario["Ограничение"])
    unavailable_transfer = options[
        (options["Вариант"] == "Межрегиональный трансфер из свободного запаса")
        & (options["Операция, ед."] == 0)
    ]
    if not unavailable_transfer.empty:
        st.caption(f"Трансфер не предлагается: {unavailable_transfer.iloc[0]['Ограничение']}.")

    result = optimize_transfer(data, event, import_day)
    if result["status"] == "optimal":
        option = result["option"]
        if option["Вариант"] == "Без вмешательства" and option["Дефицит, ед."] == 0:
            st.success(
                f"Рекомендация: без вмешательства. Для выбранного SKU/БЕ запас РЦ "
                f"{option['Запас РЦ на старте, ед.']} ед. покрывает потенциальный "
                f"спрос {option['Спрос за 28 дней, ед.']} ед. на 28-дневном горизонте. "
                "Это покрытие одного кейса, не общий KPI пилота."
            )
        else:
            st.success(result["message"])
        st.markdown(f"**Рекомендуемый вариант: {option['Вариант']}**")
        st.write(
            f"Покрытие спроса: **{option['Выполнение спроса, %']:.1f}%** · "
            f"дефицит: **{option['Дефицит, ед.']} ед.** · "
            f"модельные затраты: **{option['Итого, ₽']:,.0f} ₽**".replace(",", " ")
        )
    elif result["status"] == "unavailable":
        st.warning(result["message"])
    else:
        st.error(result["message"])


def _render_flow(data: dict[str, pd.DataFrame]) -> None:
    st.title("Поток поставки и правила решений")
    st.caption("Рычаги S&OE ограничены текущим этапом партии; объём закупки остаётся в контуре S&OP.")

    st.subheader("Путь товара и зоны управления")
    stages = [
        ("Заявки и заказ", "Прогноз 4–5 мес."),
        ("Производство", "30–45 дней"),
        ("Готово к отгрузке", "0–14 дней"),
        ("Море или ж/д", "45–90 дней"),
        ("Таможня", "7–14 дней"),
        ("Центральный РЦ", "3–7 дней"),
        ("Региональный РЦ", "3–10 дней"),
        ("Магазин", "Продажа"),
    ]
    stage_cards = "".join(
        f'<div class="soe-flow-step"><i>Этап {index:02}</i><b>{name}</b><small>{duration}</small></div>'
        for index, (name, duration) in enumerate(stages, start=1)
    )
    st.markdown(
        '<div class="soe-flow-zones"><span class="soe-flow-sop">S&OP · план закупок</span>'
        '<span class="soe-flow-soe">S&OE · срок, маршрут, распределение</span>'
        '<span class="soe-flow-sales">Продажи</span></div>'
        f'<div class="soe-flow-steps">{stage_cards}</div>',
        unsafe_allow_html=True,
    )

    stage_order = ["ready_not_shipped", "sea", "rail", "port", "customs", "received_dc"]
    shipment_counts = data["shipments"].groupby("stage").size().reindex(stage_order, fill_value=0)
    stage_counts = go.Figure(go.Bar(
        x=shipment_counts.tolist(),
        y=[STAGE_LABELS[stage] for stage in stage_order],
        orientation="h",
        marker={"color": [AMBER if stage in {"port", "customs"} else INDIGO for stage in stage_order]},
        text=shipment_counts.tolist(),
        textposition="outside",
        hovertemplate="%{y}: %{x} партий<extra></extra>",
    ))
    stage_counts.update_layout(xaxis_title="Число партий", yaxis={"autorange": "reversed"})
    st.subheader("Партии в пути по этапам")
    st.plotly_chart(_chart_layout(stage_counts, height=300), width="stretch", config={"displayModeBar": False})

    st.subheader("Заморозка решений по этапам")
    for _, zone in data["freeze_zones"].iterrows():
        stage_names = ", ".join(STAGE_LABELS[stage] for stage in zone["stages"])
        actions = ", ".join(LEVER_LABELS[lever] for lever in zone["levers"])
        decision_level = DECISION_LEVEL_LABELS[zone["decision_level"]]
        with st.expander(f"{zone['zone']} · {stage_names} · Решает: {decision_level}"):
            st.write(f"**Допустимые меры:** {actions}")
            if zone["zone"] == "Z3_FACTORY":
                st.success("Уровень решения: S&OP. Изменение объёма заказа и отмена не входят в недельный S&OE.")

    st.subheader("Недельный ритм")
    cadence = pd.DataFrame([
        {"День": "Понедельник", "Работа": "Обновление данных и панели проблем", "Владелец": "Команда данных"},
        {"День": "Вторник", "Работа": "Разбор дефицитов и заявок регионов", "Владелец": "Торговые БЕ"},
        {"День": "Среда", "Работа": "Сроки прибытия и меры по импорту", "Владелец": "БЕ «Импорт»"},
        {"День": "Четверг", "Работа": "Только неразрешённые споры, до 8 случаев", "Владелец": "Нейтральный координатор"},
        {"День": "Пятница", "Работа": "Исполнение решений и контроль", "Владелец": "Владельцы решений"},
    ])
    st.dataframe(cadence, width="stretch", hide_index=True)

    _render_allocation_rules(data)


def _render_allocation_rules(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Общие правила заявок и распределения")
    rule_cols = st.columns(3)
    releases = data["request_releases"]
    released_qty = int(releases["qty"].sum()) if not releases.empty else 0
    refusal_compensation = int(releases["compensation_rub"].sum()) if not releases.empty else 0
    rule_cols[0].metric("Отказ от firm-заявки", "Компенсируемый")
    rule_cols[0].caption(f"В примере: высвобождено {released_qty} ед.; компенсация {refusal_compensation:,.0f} ₽.".replace(",", " "))
    rule_cols[1].metric("Сверх заявок", "Не более 10%")
    rule_cols[1].caption("Лимит применяется к закупочному объёму импорта и утверждается в месячном S&OP.")
    rule_cols[2].metric("Дефицит", "Равное покрытие")
    rule_cols[2].caption("Ограниченный лот направляется туда, где он выравнивает запас в днях между дивизионами.")

    with st.expander("Симулятор распределения дефицитного лота"):
        trade_units = data["business_units"].query("bu_type == 'trade'")
        unit_names = dict(zip(trade_units["bu_id"], trade_units["name"]))
        request_skus = data["firm_requests"]["sku"].drop_duplicates().tolist()
        selected_sku = st.selectbox("SKU с firm-заявками", request_skus, key="allocation_sku")
        available_qty = st.slider("Доступно в поступившем лоте, ед.", 0, 3000, 600, 50, key="allocation_qty")

        stock_rows = data["inventory"][data["inventory"]["sku"] == selected_sku].copy()
        stock_rows["free_qty"] = (stock_rows["on_hand"] - stock_rows["reserved"]).clip(lower=0)
        stock_by_bu = stock_rows.groupby("bu_id")["free_qty"].sum().to_dict()
        sales = data["sales_history"][data["sales_history"]["sku"] == selected_sku]
        recent = sales.groupby("bu_id").tail(4)
        daily_demand = (recent.groupby("bu_id")["potential_demand"].mean() / 7).to_dict()
        requests = data["firm_requests"]
        requests = requests[(requests["sku"] == selected_sku) & requests["bu_id"].isin(trade_units["bu_id"])]
        request_caps = requests.set_index("bu_id")["qty"].astype(int).to_dict()
        allocation = allocate_equal_cover_days(available_qty, stock_by_bu, daily_demand, request_caps)
        allocation["Бизнес-единица"] = allocation["bu_id"].map(unit_names)
        allocation = allocation.rename(columns={
            "stock_before": "Запас до поставки, ед.",
            "daily_demand": "Потенциальный спрос, ед./день",
            "allocation_qty": "Выделено из лота, ед.",
            "coverage_days_after": "Запас после решения, дней",
        })
        total_allocated = int(allocation["Выделено из лота, ед."].sum())
        st.caption(f"Алгоритм распределил {total_allocated} из {available_qty} ед.; остаток лота не назначается без firm-потребности.")
        st.dataframe(
            allocation[["Бизнес-единица", "Запас до поставки, ед.", "Потенциальный спрос, ед./день", "Выделено из лота, ед.", "Запас после решения, дней"]],
            width="stretch",
            hide_index=True,
        )

    with st.expander("Проверка лимита дополнительного импорта"):
        input_cols = st.columns(2)
        confirmed_qty = input_cols[0].number_input("Подтверждённые firm-заявки, ед.", min_value=0, value=1000, step=100)
        requested_extra = input_cols[1].number_input("Запрошено сверх заявок, ед.", min_value=0, value=250, step=50)
        allowed_extra = cap_extra_import_quantity(confirmed_qty, requested_extra)
        st.info(f"При базе {confirmed_qty:,} ед. максимум сверх заявок: {int(confirmed_qty * 0.1):,} ед. Разрешённый объём из запроса: {allowed_extra:,} ед.".replace(",", " "))


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
    stock_by_category["Короткая категория"] = stock_by_category["category_id"].map({
        "CAT-DISH": "Посуда / кухня",
        "CAT-SDA": "МБТ",
        "CAT-DECOR": "Декор",
    })

    left, right = st.columns(2)
    with left:
        st.subheader("Продажи и потенциальный спрос")
        sales_figure = go.Figure()
        sales_figure.add_trace(go.Scatter(
            x=sales_trend["week_start"], y=sales_trend["Потенциальный спрос"],
            name="Потенциальный спрос", mode="lines", line={"color": MID_BLUE, "width": 2, "dash": "dot"},
        ))
        sales_figure.add_trace(go.Scatter(
            x=sales_trend["week_start"], y=sales_trend["Факт продаж"],
            name="Факт продаж", mode="lines", line={"color": NAVY, "width": 3},
            fill="tozeroy", fillcolor="rgba(89,105,198,0.10)",
        ))
        sales_figure.update_layout(yaxis_title="Единиц", hovermode="x unified")
        st.plotly_chart(_chart_layout(sales_figure, height=340), width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Остаток против страхового запаса")
        stock_figure = go.Figure()
        stock_figure.add_trace(go.Bar(y=stock_by_category["Короткая категория"], x=stock_by_category["Доступный остаток"], name="Доступный остаток", orientation="h", marker_color=INDIGO))
        stock_figure.add_trace(go.Bar(y=stock_by_category["Короткая категория"], x=stock_by_category["Страховой запас"], name="Страховой запас", orientation="h", marker_color=AMBER))
        stock_figure.update_layout(barmode="group", xaxis_title="Единиц", yaxis={"autorange": "reversed", "automargin": True})
        st.plotly_chart(_chart_layout(stock_figure, height=340), width="stretch", config={"displayModeBar": False})

    events = _prepare_events(data, selected_bu, selected_category)
    active_events = events[~events["status"].isin(["decided", "executed"])]
    priority_order = ["Критический", "Высокий", "Средний", "Низкий"]
    priority_counts = (
        active_events.groupby("priority").size().reindex(priority_order, fill_value=0)
        .rename_axis("Приоритет").reset_index(name="Событий")
    )
    st.subheader("Открытые проблемы по приоритету")
    priority_figure = go.Figure(go.Bar(
        x=priority_counts["Приоритет"], y=priority_counts["Событий"],
        marker_color=["#D85B50", AMBER, INDIGO, MID_BLUE],
        text=priority_counts["Событий"], textposition="outside",
        hovertemplate="%{x}: %{y} событий<extra></extra>",
    ))
    priority_figure.update_layout(yaxis_title="Число событий")
    st.plotly_chart(_chart_layout(priority_figure, height=300), width="stretch", config={"displayModeBar": False})
    st.caption("Доступность в графиках — среднее значение синтетического in-stock share, не целевой KPI пилота.")


def _render_pilot_targets() -> None:
    st.title("Цели и критерии пилота")
    st.caption("Ориентиры из питч-дека. Это целевые значения проекта, а не факт синтетической модели.")
    st.metric("Ожидаемый годовой эффект", "≈ 1,3% выручки импорта")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Экономический эффект")
        st.caption("млн ₽ в год на каждые 10 млрд ₽ выручки импорта · оценка из презентации")
        effect = go.Figure(go.Waterfall(
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Рост наличия", "Меньше уценок", "Оборотный капитал", "Меньше срочных мер", "Итого"],
            y=[42, 40, 40, 5, 0],
            text=["42", "40", "40", "5", "127"],
            textposition="outside",
            connector={"line": {"color": GRID, "width": 1}},
            increasing={"marker": {"color": MID_BLUE}},
            totals={"marker": {"color": NAVY}},
            hovertemplate="%{x}: %{y} млн ₽<extra></extra>",
        ))
        effect.update_layout(yaxis_title="млн ₽ в год", showlegend=False)
        st.plotly_chart(_chart_layout(effect, height=350), width="stretch", config={"displayModeBar": False})
    with right:
        st.subheader("Прогноз эффекта пилота")
        st.markdown(
            "**Рост наличия** — главный эффект, обязательный для приёмки.\n\n"
            "Снижение уценок и ускорение оборота высвобождают деньги; меньше срочных мер снижает дополнительные расходы.\n\n"
            "Оценка модельная: проверяется сравнением с контрольными категориями и прошлым сезоном."
        )

    st.subheader("Риски внедрения")
    risk_names = [
        "Сопротивление регионов",
        "Неточные сроки ETA",
        "Перегрузка совещания",
        "Подмена S&OP циклом S&OE",
        "Эффект пилота теряется в шуме",
    ]
    risk_x = [3, 3, 2, 1, 2]
    risk_y = [4, 2, 3, 4, 1]
    matrix = go.Figure()
    risk_fills = ["#FFFFFF", "#FFF7E8", "#FFE8B5", "#FFD27A"]
    for probability in range(1, 5):
        for impact in range(1, 5):
            intensity = min(3, (probability * impact - 1) // 4)
            matrix.add_shape(
                type="rect", x0=probability - 0.5, x1=probability + 0.5,
                y0=impact - 0.5, y1=impact + 0.5,
                line={"color": "#FFFFFF", "width": 2}, fillcolor=risk_fills[intensity],
                layer="below",
            )
    matrix.add_trace(go.Scatter(
        x=risk_x, y=risk_y, mode="markers+text",
        text=[str(index) for index in range(1, 6)], textposition="middle center",
        marker={"size": 28, "color": NAVY, "line": {"color": "#FFFFFF", "width": 2}},
        customdata=risk_names,
        hovertemplate="%{customdata}<br>Вероятность: %{x}<br>Влияние: %{y}<extra></extra>",
        showlegend=False,
    ))
    matrix.update_layout(
        xaxis={"title": "Вероятность", "tickvals": [1, 2, 3, 4], "ticktext": ["Низкая", "Средняя", "Высокая", "Очень высокая"], "range": [0.5, 4.5], "showgrid": False},
        yaxis={"title": "Влияние", "tickvals": [1, 2, 3, 4], "ticktext": ["Низкое", "Среднее", "Высокое", "Критическое"], "range": [0.5, 4.5], "showgrid": False},
    )
    left, right = st.columns([1, 1.2])
    with left:
        st.plotly_chart(_chart_layout(matrix, height=380), width="stretch", config={"displayModeBar": False})
    with right:
        mitigations = pd.DataFrame([
            {"№": 1, "Риск": risk_names[0], "Мера": "Правила утверждаются S&OP; отказ от заявки дешевле до отгрузки"},
            {"№": 2, "Риск": risk_names[1], "Мера": "На пилоте ETA вводится вручную, затем подключается API"},
            {"№": 3, "Риск": risk_names[2], "Мера": "На встречу выносится не более 8 споров"},
            {"№": 4, "Риск": risk_names[3], "Мера": "Объёмы и бюджеты остаются в месячном S&OP"},
            {"№": 5, "Риск": risk_names[4], "Мера": "Сравнение с контрольными категориями и сезоном"},
        ])
        st.dataframe(mitigations, width="stretch", hide_index=True)

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
    timeline = go.Figure(go.Bar(
        x=[3, 6, 5, 6],
        base=[0.5, 3.5, 9.5, 12.5],
        y=["Диагностика", "Пилот", "Тиражирование", "Автоматизация"],
        orientation="h",
        marker_color=[MUTED, INDIGO, NAVY, AMBER],
        text=["1–3 мес.", "4–9 мес.", "10–14 мес.", "13–18 мес."],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    timeline.update_layout(
        xaxis={"title": "Месяц проекта", "tickvals": list(range(1, 19)), "range": [0.5, 18.5], "gridcolor": GRID},
        yaxis={"autorange": "reversed"}, showlegend=False,
    )
    st.plotly_chart(_chart_layout(timeline, height=300), width="stretch", config={"displayModeBar": False})
    st.dataframe(roadmap, width="stretch", hide_index=True)
    st.subheader("Как проверяется эффект")
    comparison = pd.DataFrame([
        {"Сравнение": "Пилотные категории vs контрольные категории", "Что отделяет": "Общий рыночный и сезонный фон"},
        {"Сравнение": "Волна 1 (~50% импорта) vs волна 2", "Что отделяет": "Эффект подключения дивизионов к циклу"},
        {"Сравнение": "Текущий сезон vs прошлый сезон", "Что отделяет": "Сезонность и базовую динамику категории"},
    ])
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption("Эффект оценивается по изменению показателей, а не только по итоговым уровням; категории и волны сопоставляются до/после включения в цикл.")
    st.subheader("Приёмка")
    st.markdown(
        "- Импорт участвует в цикле не менее 3 месяцев.\n"
        "- Выполнено не менее 4 из 6 целей, причём цель по наличию обязательна.\n"
        "- Регламент, RACI и правила утверждены S&OP; панель работает штатно.\n"
        "- Зрелость S&OE достигла уровня 4 из 5.\n"
        "- Эффект измеряется сравнением с контрольными категориями и предыдущим сезоном."
    )


def run() -> None:
    _apply_visual_style()
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
