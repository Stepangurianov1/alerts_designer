import streamlit as st
from ui_test import main as constructor_main
from settings_params import main as settings_main


def main():
    # сетап только здесь
    st.set_page_config(page_title="Alert Params", layout="wide")

    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Страница",
        ["Конструктор", "Мои кастомные алерты"],
    )

    if page == "Конструктор":
        constructor_main()
    else:
        settings_main()


if __name__ == "__main__":
    main()
