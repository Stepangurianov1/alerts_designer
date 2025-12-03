from jinja2 import pass_eval_context
import pandas as pd
import streamlit as st
from sqlalchemy import text
from typing import Any, Dict
import re

from execute_sql import run_query_dwh            
from consts import DWH_ENGINE     
from ui_test import parse_numeric_params, render_message_with_config  



def load_customized_events() -> pd.DataFrame:
    """
    Загружает список всех event_id, для которых есть кастомные параметры,
    вытягивает к ним сообщение из alerts_history.
    """
    query = """
        SELECT
            p.event_id,
            p.alert_name,
            p.change_id,
            h.message,
            MAX(p.dttm_msk) AS last_change,
            BOOL_OR(COALESCE(p.is_active, TRUE)) AS is_active
        FROM alerts.alert_params p
        JOIN alerts.alerts_history h
            ON h.event_id = p.event_id
        GROUP BY p.event_id, p.alert_name, h.message, p.change_id
        ORDER BY last_change DESC;
    """
    return run_query_dwh(query)


def load_params_for_event(event_id: int, change_id: int) -> pd.DataFrame:
    """
    Загружает параметры (config) для конкретного event_id.
    """
    query = f"""
        SELECT
            param_index,
            param_number,
            mode,
            values,
            operator,
            threshold,
            is_active,
            dttm_msk
        FROM alerts.alert_params
        WHERE event_id = {event_id}
        AND change_id = {change_id}
        ORDER BY param_index;
    """
    return run_query_dwh(query)


def update_event_is_active(event_id: int, active: bool) -> None:
    """
    Включает/выключает правило на уровне всего сообщения (event_id).
    """
    query = """
        UPDATE alerts.alert_params
        SET is_active = :active
        WHERE event_id = :event_id
    """
    with DWH_ENGINE.begin() as conn:
        conn.execute(text(query), {"active": active, "event_id": event_id})


def build_config_from_db(df_params: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Переводит параметры из БД в формат config, который использует render_message.
    Пропускает строки, где param_index = NULL.
    """
    config: Dict[int, Dict[str, Any]] = {}

    for _, row in df_params.iterrows():
        if pd.isna(row["param_index"]):
            continue  # пропускаем битые/старые строки без индекса

        idx = int(row["param_index"])
        mode = row["mode"]

        if mode == "numeric":
            thr = row["threshold"]
            config[idx] = {
                "mode": "numeric",
                "operator": row["operator"],
                "threshold": float(thr) if thr is not None else None,
            }

        elif mode in ("in_list", "not_in_list"):
            vals = row["values"] if isinstance(row["values"], list) else []
            config[idx] = {
                "mode": mode,
                "values": [str(v) for v in vals],
            }

        else:
            config[idx] = {"mode": "numeric", "operator": ">", "threshold": None}

    return config


def build_params_summary(params: list[dict], config: Dict[int, Dict[str, Any]]) -> str:
    """
    Генерирует короткий текст:
    "изменено: param1 > 6 | по умолчанию: param2,param3"
    """

    changed_parts = []
    unchanged_parts = []

    for p in params:
        idx = p["index"]
        tag = f"param{p['index'] + 1}"
        cfg = config.get(idx, {})
        mode = cfg.get("mode", "numeric")

        if mode == "numeric":
            op = cfg.get("operator", ">")
            thr = cfg.get("threshold", p["value"])
            default_op = ">"
            default_thr = p["value"]

            if op != default_op or abs(float(thr) - float(default_thr)) > 1e-9:
                changed_parts.append(f"{tag} {op} {thr}")
            else:
                unchanged_parts.append(tag)

        elif mode in ("in_list", "not_in_list"):
            values = cfg.get("values") or []
            if values:
                verb = "IN" if mode == "in_list" else "NOT IN"
                changed_parts.append(f"{tag} {verb} ({', '.join(values)})")
            else:
                unchanged_parts.append(tag)

        else:
            unchanged_parts.append(tag)

    parts = []
    if changed_parts:
        parts.append("изменено: " + ", ".join(changed_parts))
    if unchanged_parts:
        parts.append("по умолчанию: " + ", ".join(unchanged_parts))

    return " | ".join(parts)


def manage_custom_alerts_page():
    events_df = load_customized_events()
    if events_df.empty:
        st.info("Пока нет кастомных алертов (ничего не сохранено в alert_params).")
        return

    with st.sidebar:
        alert_names = sorted(events_df["alert_name"].unique())
        selected_alert = st.selectbox("Фильтр по алерту", ["Все"] + alert_names)

        active_filter = st.selectbox(
            "Статус",
            ["Все", "Только активные", "Только выключенные"],
        )

    filtered = events_df.copy()
    if selected_alert != "Все":
        filtered = filtered[filtered["alert_name"] == selected_alert]

    if active_filter == "Только активные":
        filtered = filtered[filtered["is_active"] == True]
    elif active_filter == "Только выключенные":
        filtered = filtered[filtered["is_active"] == False]

    st.caption(f"Найдено сообщений: {len(filtered)}")

    for _, row in filtered.iterrows():
        event_id = int(row["event_id"])
        alert_name = row["alert_name"]
        raw_message = row["message"]
        is_active = bool(row["is_active"])
        change_id = row["change_id"]

        with st.expander(alert_name, expanded=False):

            # parameters from DB
            params_df = load_params_for_event(event_id, change_id)
            if params_df.empty:
                st.warning("Нет параметров для этого сообщения.")
                st.text_area("Сообщение", value=raw_message, height=180)
                continue

            config = build_config_from_db(params_df)
            clean_message, parsed_params = parse_numeric_params(raw_message)

            summary = build_params_summary(parsed_params, config)
            st.markdown(f"**Условия:** {summary}")

            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.markdown("**Сообщение с подстановкой условий:**")
                highlighted_html = render_message_with_config(clean_message, parsed_params, config)
                st.markdown(
                    "<div style='white-space:pre-wrap; font-size:14px;'>"
                    f"{highlighted_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with col_right:
                col_1, col_2 = st.columns([2, 1])
                with col_1:
                    st.markdown("**Настройки:**")

                    new_state = st.checkbox(
                        "Правило активно?",
                        value=is_active,
                        key=f"active_{event_id}_{change_id}",
                    )
                    if new_state != is_active:
                        update_event_is_active(event_id, new_state)
                        st.success("is_active обновлён")

                    st.caption(f"Последнее изменение: {row['last_change']}")
                with col_2:
                    if st.button('Удалить', key=f"delete_{event_id}_{change_id}"):
                        delete_query = f"""
                        DELETE FROM alerts.alert_params
                        WHERE change_id = {change_id}
                        """
                        with DWH_ENGINE.begin() as conn:
                            conn.execute(text(delete_query))
                        st.rerun()

def main():
    st.set_page_config(page_title="Custom Alerts", layout="wide")
    manage_custom_alerts_page()


