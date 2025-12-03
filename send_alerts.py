from datetime import datetime, timedelta
import pandas as pd
import psycopg2
import requests
from sqlalchemy import create_engine
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.hooks.base_hook import BaseHook
from parameters_parsing import parse_numeric_params
# from telegram_utils import notify_failure


# ===================== Airflow Connections =====================

# con_csd = BaseHook.get_connection('csd_bi_connection')
# con_aifory = BaseHook.get_connection('aifory_connection')
# con_p2p_prod = BaseHook.get_connection('aifory_prod_connection')
# con_dwh = BaseHook.get_connection('dwh_connection')
# con_tickets = BaseHook.get_connection('ticket_replica')


# ===================== DB Helpers =====================

def _run_query(host_con, dbname: str, query: str) -> pd.DataFrame:
    connection = psycopg2.connect(
        dbname=dbname,
        user=host_con['login'],
        password=host_con['password'],
        host=host_con['host'],
        port=host_con['port'],
    )
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        cursor.close()
        connection.close()

# def run_query_cbs_merchants(query: str) -> pd.DataFrame:
#     return _run_query(con_csd, "cbs_replica", query)

def run_query_csd(query: str) -> pd.DataFrame:
    con_csd = {
        "host": "138.68.88.175",
        "port": 5432,
        "name": "csd_bi",
        "login": "datalens_utl",
        "password": "fNL60YHONhaxa1AJ7Onhpq"
    }
    return _run_query(con_csd, "csd_bi", query)


# def run_query_ticket_replica(query: str) -> pd.DataFrame:
#     return _run_query(con_tickets, "ticket_replica", query)


# def run_query_p2p(query: str) -> pd.DataFrame:
#     return _run_query(con_aifory, "aifory_p2p_prod", query)


# def run_query_p2p_prod(query: str) -> pd.DataFrame:
#     return _run_query(con_p2p_prod, "aifory_p2p_prod", query)


# def run_query_p2p_gateway(query: str) -> pd.DataFrame:
#     return _run_query(con_p2p_prod, "p2p_payout_gateway", query)


def run_query_dwh(query: str) -> pd.DataFrame:
    con_dwh = {
        "host": "49.12.21.243",
        "port": 6432,
        "name": "postgres",
        "login": "second_bi_user",
        "password": "sdsdGVGYJ12"
    }
    return _run_query(con_dwh, "postgres", query)


# ===================== Telegram =====================

def send_message_bot(text: str, bot_token: str, chat_id: int, thread_id: int | None) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    if thread_id is not None:
        payload["message_thread_id"] = thread_id

    response = requests.post(url, params=payload)

    if response.status_code == 200:
        print("Сообщение отправлено успешно!")
    else:
        print("Ошибка при отправке:", response.text)



# ===================== Правила из Streamlit (alert_params) =====================

def load_all_alert_params() -> pd.DataFrame:
    """
    Забираем ВСЕ активные, изменённые правила.
    Никаких groupby — нам важны все change_id.
    """
    query = """
        SELECT
            alert_name,
            param_index,
            param_number,
            mode,
            values,
            operator,
            threshold,
            is_active,
            is_changed,
            change_id
        FROM alerts.alert_params
        WHERE is_active = TRUE
          AND is_changed = TRUE
    """
    return run_query_dwh(query)


def check_message_against_config(parsed_params: list[dict], config: dict[int, dict]) -> bool:
    """
    parsed_params — результат parse_numeric_params(message)[1]
    config — { param_index: {mode, operator/values, threshold} }
    Возвращает True, если сообщение удовлетворяет ВСЕМ правилам этого конфига.
    """
    for p in parsed_params:
        idx = p["index"]
        if idx not in config:
            continue

        rule = config[idx]
        mode = rule.get("mode")
        value = p["value"]

        if mode == "numeric":
            op = rule.get("operator")
            thr = rule.get("threshold")

            if thr is None or op is None:
                continue

            if op == ">" and not (value > thr):
                return False
            if op == "<" and not (value < thr):
                return False
            if op == ">=" and not (value >= thr):
                return False
            if op == "<=" and not (value <= thr):
                return False
            if op == "==" and not (value == thr):
                return False
            if op == "!=" and not (value != thr):
                return False

        elif mode in ("in_list", "not_in_list"):
            vals = rule.get("values") or []
            vals_str = {str(v) for v in vals}
            if float(value).is_integer():
                v_str = str(int(value))
            else:
                v_str = str(value)

            if mode == "in_list" and v_str not in vals_str:
                return False
            if mode == "not_in_list" and v_str in vals_str:
                return False

    return True


def apply_custom_filters(message: str, params_df: pd.DataFrame) -> bool:
    """
    params_df — все строки из alerts.alert_params для КОНКРЕТНОГО alert_name.
    Логика:
      - группы по change_id = разные наборы правил;
      - сообщение проходит, если выполняется ХОТЯ БЫ один набор (OR по change_id);
      - внутри набора все правила по param_index — AND.
    """
    if params_df.empty:
        return True

    _, parsed_params = parse_numeric_params(message)
    if not parsed_params:
        return True

    if "change_id" not in params_df.columns or params_df["change_id"].isna().all():
        groups = {"_single": params_df}
    else:
        groups = {cid: sub for cid, sub in params_df.groupby("change_id")}

    for cid, sub_df in groups.items():
        config: dict[int, dict] = {}
        for _, row in sub_df.iterrows():
            idx = int(row["param_index"])
            mode = row["mode"]

            if mode == "numeric":
                config[idx] = {
                    "mode": "numeric",
                    "operator": row["operator"],
                    "threshold": row["threshold"],
                }
            elif mode in ("in_list", "not_in_list"):
                vals = row["values"] if isinstance(row["values"], list) else []
                config[idx] = {
                    "mode": mode,
                    "values": vals,
                }

        if check_message_against_config(parsed_params, config):
            return True

    return False


# ===================== Основная логика отправки алертов =====================

def send_alert():
    chat_settings = run_query_dwh(
        """
        SELECT name, chat_id, thread_id, bot_token, geo, project_id
        FROM alerts.chat_settings
        """
    )

    alert_sample = run_query_dwh(
        """
        SELECT *
        FROM alerts.alert_sample
        where id = 54
        """
    )

    # уже отправленные за последние 20 минут (чтобы не дублировать)
    cur_alerts = run_query_dwh(
        """
        SELECT alert_name || '_' || COALESCE(id::text, 'all') || '_' || geo AS field
        FROM alerts.alerts_history
        WHERE etl_dttm >= now() - interval '20 minutes'
        """
    )
    cur_alerts_lst = cur_alerts["field"].to_list()

    # все правила из Streamlit
    all_params = load_all_alert_params()

    alerts = alert_sample["id"].to_list()
    cur_alert_df = pd.DataFrame(
        columns=["alert_name", "id", "geo", "value", "etl_dttm", "message", "project_id"]
    )

    for j in alerts:
        cur_alert = alert_sample[alert_sample["id"] == j].reset_index(drop=True)
        if not bool(cur_alert["is_active"][0]):
            continue

        print(f"Обработка алерта: {cur_alert['alert_name'][0]}")

        query = cur_alert["sql_code"][0]
        db_name = cur_alert["db_name"][0]

        if db_name == "cascade":
            df = run_query_csd(query)
        # elif db_name == "p2p":
        #     df = run_query_p2p(query)
        # elif db_name == "dwh":
        #     df = run_query_dwh(query)
        # elif db_name == "cbs_merchant":
        #     df = run_query_cbs_merchants(query)
        # elif db_name == "ticket_replica":
        #     df = run_query_ticket_replica(query)
        else:
            print(f"Неизвестный db_name: {db_name}, пропускаем")
            continue

        if df.empty:
            continue

        if "project_id" not in df.columns:
            df["project_id"] = 1

        df["alert_name"] = cur_alert["alert_name"][0]
        df["name"] = cur_alert["type"][0]

        # здесь считаем, что SQL уже вернул только те строки, по которым нужна тревога

        df = df.merge(
            chat_settings,
            on=["name", "geo", "project_id"],
            how="left",
        )

        # сравнение с прошлыми значениями (если настроено)
        if cur_alert["hours_compare_with_past"][0] is not None and pd.notna(
            cur_alert["hours_compare_with_past"][0]
        ):
            hours_compare_with_past = cur_alert["hours_compare_with_past"][0]
            alert_name = cur_alert["alert_name"][0]

            compare_with_past_df = run_query_dwh(
                f"""
                SELECT DISTINCT ON (alert_name, id, geo)
                    alert_name,
                    id,
                    geo,
                    REGEXP_REPLACE(value, '[^0-9\.-]', '', 'g')::numeric AS old_value,
                    COALESCE(project_id, 1) AS project_id
                FROM alerts.alerts_history
                WHERE etl_dttm >= now() - interval '{hours_compare_with_past} hours'
                  AND alert_name = '{alert_name}'
                ORDER BY alert_name, id, geo, etl_dttm DESC
                """
            )

            df = df.merge(
                compare_with_past_df,
                how="left",
                on=["alert_name", "id", "geo", "project_id"],
            )

            if "old_value" in df.columns and "threshold" in df.columns:
                df = df[
                    (df["old_value"].isna())
                    | (df["threshold"] > df["old_value"])
                ]

        if df.empty:
            continue

        print("Готовых к отправке строк:", df.shape[0])

        for i in range(df.shape[0]):
            mes_template = cur_alert["message"][0]
            mes_template = mes_template.replace("\\n", "\n").replace("\\t", "\t")

            # защищённо берём поля, если они есть
            def _safe(col: str, default=""):
                return df[col].iloc[i] if col in df.columns else default

            message = mes_template.format(
                id=_safe("id"),
                geo=_safe("geo"),
                benchmark=_safe("benchmark"),
                threshold=_safe("threshold"),
                meta1=_safe("meta1"),
                meta2=str(_safe("meta2")).replace("\\n", "\n"),
                tags=_safe("tags"),
            )

            field_key = (
                f"{cur_alert['alert_name'][0]}_{_safe('id')}_{_safe('geo')}"
            )
            if field_key in cur_alerts_lst:
                continue

            # применяем кастомные фильтры из Streamlit, если они есть
            alert_name = cur_alert["alert_name"][0]
            if not all_params.empty:
                params_df = all_params[all_params["alert_name"] == alert_name]
                if not params_df.empty:
                    if not apply_custom_filters(message, params_df):
                        print(f"⛔ Сообщение отфильтровано правилами: {alert_name}")
                        continue

            cur_alert_df = pd.concat(
                [
                    cur_alert_df,
                    pd.DataFrame(
                        [
                            [
                                cur_alert["alert_name"][0],
                                _safe("id"),
                                _safe("geo"),
                                _safe("threshold"),
                                datetime.now(),
                                message,
                                _safe("project_id"),
                            ]
                        ],
                        columns=[
                            "alert_name",
                            "id",
                            "geo",
                            "value",
                            "etl_dttm",
                            "message",
                            "project_id",
                        ],
                    ),
                ],
                ignore_index=True,
            )

            send_message_bot(
                message,
                _safe("bot_token"),
                _safe("chat_id"),
                _safe("thread_id"),
            )

    if not cur_alert_df.empty:
        engine = create_engine(
            "postgresql://{user}:{password}@{host}:{port}/postgres".format(
                user="second_bi_user",
                password="sdsdGVGYJ12",
                host="49.12.21.243",
                port=6432
            )
        )
        cur_alert_df.to_sql(
            schema="alerts",
            name="alerts_history",
            if_exists="append",
            con=engine,
            index=False,
        )

if __name__ == '__main__':
    send_alert()

# default_args = {
#     "owner": "smallboy",
#     "retries": 1,
#     "retry_delay": timedelta(seconds=20),
#     "on_failure_callback": notify_failure,
# }

# with DAG(
#     dag_id="processing_alert",
#     default_args=default_args,
#     schedule_interval="*/5 * * * *",
#     start_date=datetime(2025, 6, 15),
#     catchup=False,
# ) as dag:
#     task_send_alert = PythonOperator(
#         task_id="send_alert",
#         python_callable=send_alert,
#     )
