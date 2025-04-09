import streamlit as st
import explain


def run():
    st.title("Local Explanation")
    approach = st.selectbox(
        "Explaining Approach:", ["SHAP", "LIME", "Integrated Gradients"]
    )
    text = st.text_input("Sample to be explained:")
    if st.button("explain"):
        explain.explain(text, approach)


if __name__ == "__main__":
    run()
