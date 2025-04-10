import streamlit as st


def main():
    st.set_page_config(
        page_title="Explanation Helper", page_icon=":guardsman:", layout="centered"
    )
    # st.title("Explanation Helper")
    home_page = st.Page(r"pages/p_home.py", title="Homepage")
    eda_page = st.Page(r"pages/p_eda.py", title="EDA")
    clss_eval_page = st.Page(r"pages/p_clss_eval.py", title="Classification Evaluation")
    # expl_eval_page = st.Page(r"pages/p_expl_eval.py", title="Explanation Evaluation")
    expl_page = st.Page(r"pages/p_expl.py", title="Local Explanation")
    find_wrong_page = st.Page(r"pages/p_find_mismatch.py", title="Find Wrong Predictions")

    pg = st.navigation(
        [home_page, eda_page, clss_eval_page, expl_page, find_wrong_page]
    )

    pg.run()


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "main"
    main()
