import re
import streamlit as st
import pandas as pd
from execute_sql import get_messages
from datetime import datetime
import pytz
from consts import DWH_ENGINE


def strip_html_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+%", "%", text)
    return text


def parse_numeric_params(message: str, context_window: int = 40):
    clean_message = strip_html_tags(message)
    clean_message = clean_message.replace("\r\n", "\n").replace("\r", "\n")

    IGNORE_PATTERNS = [
        r"\b\d{4}-\d{2}-\d{2}\b",          # 2025-12-02
        r"\b\d{2}\.\d{2}\.\d{4}\b",        # 02.12.2025
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",   # 12:55 или 12:55:30
        r"\b\d{4}/\d{2}/\d{2}\b",          # 2025/12/02
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",    # 2-12-2025
    ]

    ignore_spans: list[tuple[int, int]] = []
    for pat in IGNORE_PATTERNS:
        for m in re.finditer(pat, clean_message):
            ignore_spans.append(m.span())

    def is_inside_ignored(start: int, end: int) -> bool:
        for s, e in ignore_spans:
            if start >= s and end <= e:
                return True
        return False

    number_pattern = re.compile(
        r"(?:(?<=\s)|(?<=\())(-?\d+(?:\.\d+)?)(\s*%)?"
        r"|"
        r"^(-?\d+(?:\.\d+)?)(\s*%)?"
    )

    params: list[dict] = []

    for i, match in enumerate(number_pattern.finditer(clean_message)):
        if match.group(1) is not None:
            value_str = match.group(1)
            percent_sign = match.group(2)
            start, end = match.span(1)
        else:
            value_str = match.group(3)
            percent_sign = match.group(4)
            start, end = match.span(3)

        if is_inside_ignored(start, end):
            continue

        try:
            value = float(value_str)
        except ValueError:
            continue

        is_percent = percent_sign is not None and "%" in percent_sign

        # контекст = вся строка, где находится число
        line_start = clean_message.rfind("\n", 0, start)
        if line_start == -1:
            line_start = 0
        else:
            line_start += 1  

        line_end = clean_message.find("\n", end)
        if line_end == -1:
            line_end = len(clean_message)

        context = clean_message[line_start:line_end]

        params.append(
            {
                "index": i,
                "tag": f"param {i + 1}",
                "value": value,
                "is_percent": is_percent,
                "raw": clean_message[start:end],
                "start": start,
                "end": end,
                "context": context.strip(),
            }
        )

    return clean_message, params



def render_message_with_config(clean_message: str, params: list, config: dict):
    """
    Оборачиваем числа:
      — по умолчанию только 'param N'
      — если выбрано числовое условие и оно изменено: 'param N: >= 4.0'
      — если 'в списке/не в списке' и список не пуст: 'param N: в списке (1,2,3)'
    """
    if not params:
        return clean_message
    new_params = {}
    result_parts = []
    last_pos = 0

    for p in params:

        start, end = p["start"], p["end"]
        result_parts.append(clean_message[last_pos:start])

        num_text = clean_message[start:end]
        tag = p["tag"]

        cfg = config.get(p["index"], {}) or {}
        mode = cfg.get("mode", "numeric")  # numeric / in_list / not_in_list

        label_text = tag
        changed = False

        if mode == "numeric":
            op = cfg.get("operator")
            thr = cfg.get("threshold")
            default_op = ">"
            default_thr = p["value"]

            if (
                op is not None
                and thr is not None
                and (op != default_op or abs(float(thr) - float(default_thr)) > 1e-9)
            ):
                label_text = f"{tag}: {op} {thr}"
                changed = True

        elif mode in ("in_list", "not_in_list"):
            values = cfg.get("values") or []
            if values:
                verb = "в списке" if mode == "in_list" else "не в списке"
                label_text = f"{tag}: {verb} ({', '.join(values)})"
                changed = True

        bg = "#ffe082" if changed else "#fff9c4"

        span_html = (
            f"<span style='background:{bg}; padding:2px 4px; "
            f"border-radius:3px; color:#000;'>"
            f"<b>{num_text}</b> "
            f"<span style='color:#000;font-size:0.8em;'>{label_text}</span>"
            f"</span>"
        )

        result_parts.append(span_html)
        last_pos = end

    result_parts.append(clean_message[last_pos:])
    return "".join(result_parts)

def insert_sessions():
    if 'alert_messages_df' not in st.session_state:
        st.session_state['alert_messages_df'] = get_messages()



def config_dict_to_df(config: dict, params: list, alert_name: str, event_id: int) -> pd.DataFrame:
    rows = []

    msk_now = datetime.now(pytz.timezone("Europe/Moscow"))

    for idx, entry in config.items():
        idx = int(idx)
        mode = entry.get("mode")
        param_context = ""
        raw_value = ""
        for p in params:
            if p["index"] == idx:
                param_context = p.get("context", "")
                raw_value = p.get("raw", "")
                break

        new_value = raw_value  

        if mode == "numeric":
            thr = entry.get("threshold")
            if thr is not None:
                new_value = str(thr)

        elif mode in ("in_list", "not_in_list"):
            vals = entry.get("values", [])
            new_value = "(" + ", ".join(vals) + ")"

        if raw_value and param_context:
            safe_raw = re.escape(raw_value)
            param_context = re.sub(safe_raw, new_value, param_context, count=1)
        row = {
            "index": idx,
            "param_number": idx + 1,
            "mode": mode,
            "values": None,
            "operator": None,
            "threshold": None,
            "alert_name": alert_name,
            "context": param_context,  
            "dttm_msk": msk_now,
            "event_id": event_id
        }

        if mode in ("in_list", "not_in_list"):
            row["values"] = entry.get("values", [])

        elif mode == "numeric":
            row["operator"] = entry.get("operator")
            row["threshold"] = entry.get("threshold")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("index").sort_index()
    return df




def main():
    st.set_page_config(page_title="Alert Params Demo", layout="wide")
    insert_sessions()
    st.sidebar.header("Алерт")

    alert_names = st.session_state['alert_messages_df']['alert_name'].tolist()
    alert_idx = st.sidebar.selectbox(
        "Выбери алерт",
        range(len(st.session_state['alert_messages_df']['message'].tolist())),
        format_func=lambda i: alert_names[i],
    )
    raw_message = st.session_state['alert_messages_df']['message'].iloc[alert_idx]
    event_id = st.session_state['alert_messages_df']['event_id'].iloc[alert_idx]

    clean_message, params = parse_numeric_params(raw_message)
    if not params:
        st.warning("В сообщении не найдено числовых параметров.")
        st.text_area("Сообщение", value=clean_message, height=300)
        return

    col_left, col_right = st.columns([2, 1])

    operators = [">", "<", ">=", "<=", "==", "!="]
    config: dict[int, dict] = {}
    with col_right:
        st.subheader("Параметры")

        for p in params:
            with st.expander(f"{p['tag']} — {p['raw']}"):
                st.markdown(f"**Контекст:** …{p['context']}…")

                mode = st.selectbox(
                    "Тип условия",
                    ["Числовое сравнение", "В списке", "Не в списке"],
                    key=f"mode_{alert_idx}_{p['index']}",
                )

                c1, c2, c3 = st.columns([1, 1, 1])

                if mode == "Числовое сравнение":
                    with c1:
                        op = st.selectbox(
                            "Оператор",
                            options=operators,
                            index=0,
                            key=f"op_{alert_idx}_{p['index']}",
                        )

                    with c2:
                        val = st.number_input(
                            "Порог",
                            value=p["value"],
                            key=f"val_{alert_idx}_{p['index']}",
                        )

                    with c3:
                        st.write("Тип:")
                        st.write("**Процент**" if p["is_percent"] else "**Число**")

                    config[p["index"]] = {
                        "mode": "numeric",
                        "operator": op,
                        "threshold": float(val),
                    }

                else:
                    with c1:
                        list_raw = st.text_input(
                            "Список значений (через запятую)",
                            key=f"list_{alert_idx}_{p['index']}",
                        )

                    with c2:
                        st.write("Тип:")
                        st.write("**Процент**" if p["is_percent"] else "**Число**")

                    with c3:
                        st.write("")  # пустой столбец

                    values = [v.strip() for v in list_raw.split(",") if v.strip()]
                    config[p["index"]] = {
                        "mode": "in_list" if mode == "В списке" else "not_in_list",
                        "values": values,
                    }

    with col_left:
        st.subheader("Сообщение")
        highlighted_html = render_message_with_config(clean_message, params, config)
        st.markdown(
            "<div style='white-space:pre-wrap; font-size:15px;'>"
            f"{highlighted_html}"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Save results to db"):
            result = config_dict_to_df(config, params, alert_names[alert_idx], event_id)
            result.to_sql(
                name="alert_params",
                con=DWH_ENGINE,
                schema="alerts",
                if_exists="append",   
                index=False          
            )

if __name__ == "__main__":
    main()
