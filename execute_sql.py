import time
import pandas as pd
import re
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from consts import CSD_ENGINE, DWH_ENGINE



def _run_query_with_retry(engine, query, max_retries=3, delay=2) -> pd.DataFrame:
    """
    Универсалка: выполняет запрос с ретраями при проблемах с подключением.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                return pd.read_sql_query(text(query), conn)
        except OperationalError as e:
            last_exc = e
            print(
                f"[WARN] Ошибка подключения (попытка {attempt}/{max_retries}): {e}"
            )
            if attempt < max_retries:
                time.sleep(delay)
    print("[ERROR] Не удалось выполнить запрос после нескольких попыток.")
    print(last_exc)
    return pd.DataFrame()


def run_query_csd(query: str) -> pd.DataFrame:
    return _run_query_with_retry(CSD_ENGINE, query)


def run_query_dwh(query: str) -> pd.DataFrame:
    return _run_query_with_retry(DWH_ENGINE, query)




def get_messages():
    url = 'https://docs.google.com/spreadsheets/d/161HvbPxfOq39piPh0F7ZRRUK4_gjYrnyfglRHIWpeaA/edit?gid=0#gid=0'
    csv_url = url.split('/edit')[0] + '/export?format=csv'

    custom_settings = pd.read_csv(csv_url)
    custom_settings = custom_settings[custom_settings['is_active']]

    alert_names = custom_settings['alert_name'].tolist()
    if not alert_names:
        print("Нет активных алертов в гуглшите.")
        return

    if len(alert_names) == 1:
        in_clause = f"('{alert_names[0]}')"
    else:
        in_clause = str(tuple(alert_names))

    query = f"""
        SELECT DISTINCT ON (alert_name)
               *
        FROM alerts.alerts_history
        WHERE alert_name IN {in_clause}
        ORDER BY alert_name, etl_dttm DESC;
    """

    df = run_query_dwh(query)
    df = df[df['message'].notna()]

    return df[['alert_name', 'message', 'event_id']]


            


# res = get_messages()
# res = res[res['alert_name'] == 'card_in_traffic']
# print(res['message'].iloc[0])