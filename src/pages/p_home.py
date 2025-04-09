import streamlit as st
import utils


def run():
    st.title("Homepage")
    with st.expander("Which model am I using?"):
        st.write("By default, it's **dima806/news-category-classifier-distilbert** from HuggingFace as an example. Since a model is typically too large to be directly uploaded here, " \
        "the model path is written in `config.yaml`. Modify that file if needed.")
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
