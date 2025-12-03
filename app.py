import streamlit as st
from ui_test import main as constructor_main
from settings_params import main as settings_main


def main():
    st.set_page_config(
        page_title="Alert Params",
        layout="wide",
    )

    # Левая панель — и навигация, и фильтры страниц
    st.sidebar.title("Навигация")

    page = st.sidebar.radio(
        "Страница",
        ["Конструктор", "Мои кастомные алерты"],
    )

    st.sidebar.markdown("---")

    if page == "Конструктор":
        constructor_main()
    else:
        st.sidebar.caption("Просмотр и включение/выключение кастомных правил")
        settings_main()


if __name__ == "__main__":
    main()
