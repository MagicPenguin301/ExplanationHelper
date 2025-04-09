import streamlit as st
import explain, utils

@utils.after_model_loaded
def run():
    approach = st.selectbox(
        "Explaining Approach:", ["SHAP", "LIME", "Integrated Gradients"]
    )
    text = st.text_input("Sample to be explained:")
    if st.button("explain"):
        with st.spinner("Explaining the sample..."):
            explain.explain(text, approach)


if __name__ == "__main__":
    st.title("Local Explanation")
    run()
