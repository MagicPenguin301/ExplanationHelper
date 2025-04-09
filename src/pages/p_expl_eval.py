import streamlit as st
import utils

@utils.after_model_loaded
def run():
    pass


if __name__ == "__main__":
    st.title("Explanation Evaluation")
    with st.expander("What's this?"):
        st.write("Explanation creates reliability, but they themselves must be tested." \
        " Get an overview of metrics tailored to each approach given a dataset here.")
    run()