import streamlit as st
import utils


def run():
    st.title("Homepage")
    with st.expander("Which model am I using?"):
        path = utils.config["model"]["path"]
        from_hf = utils.config["model"]["from_hf"]
        st.write(
            f"The current path is **{path}** \
            {"(HuggingFace)" if from_hf else "(a local path)"}. \
            Since a model is typically too large to be directly uploaded here, "
            "the model path is written in `config.yaml`. Modify that file if needed."
        )
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
