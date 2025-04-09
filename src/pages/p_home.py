import streamlit as st
import utils


def run():
    st.title("Homepage")
    if utils.model and utils.tokenizer:
        st.success("You have loaded the model!")
    elif st.button("Load the model"):
        with st.spinner("Loading model..."):
            try:
                utils.init_model()
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"An error occurred during model loading: {e}")


if __name__ == "__main__":
    run()
