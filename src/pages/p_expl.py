import streamlit as st
import explain, utils


def run():
    st.title("Local Explanation")
    if not utils.model or not utils.tokenizer:
        st.warning("The model has not been loaded!")
        st.stop()
    approach = st.selectbox(
        "Explaining Approach:", ["SHAP", "LIME", "Integrated Gradients"]
    )
    text = st.text_input("Sample to be explained:")
    if st.button("explain"):
        with st.spinner("Explaining the sample..."):
            explain.explain(text, approach)


if __name__ == "__main__":
    run()
