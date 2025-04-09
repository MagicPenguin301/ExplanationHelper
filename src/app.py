import streamlit as st

# Function to navigate to a different page
def navigate_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# def render_back_button():
#     if 'page' in st.session_state and st.session_state.page != 'main':
#         if st.button('Back'):
#             st.session_state.page = 'main'
#             st.rerun()


def main():
    st.set_page_config(page_title="Explanation Helper", page_icon=":guardsman:", layout="centered")
    # st.title("Explanation Helper")
    clss_eval_page = st.Page(r"pages/p_clss_eval.py", title="Classification Evaluation")
    eda_page = st.Page(r"pages/p_eda.py", title="EDA")
    expl_eval_page = st.Page(r"pages/p_expl_eval.py", title="Explanation Evaluation")
    expl_page = st.Page(r"pages/p_expl.py", title="Local Explanation")
    
    pg = st.navigation([clss_eval_page, eda_page, expl_eval_page, expl_page])
    
    pg.run()

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    main()
