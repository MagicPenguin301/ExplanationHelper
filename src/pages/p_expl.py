import streamlit as st
import explain, utils

@utils.after_model_loaded
def run():
    approach = st.selectbox(
        "Explaining Approach:", ["SHAP", "LIME", "Saliency"]
    )
    label_i = None
    if approach in ["LIME", "Saliency"]:
        label_str = st.selectbox("Select a label to be explained:", utils.model.config.id2label.values())
        label_i = utils.model.config.label2id[label_str]
    text = st.text_area("Sample to be explained:")
    show_infi = st.checkbox("Infidelity", False)
    if st.button("explain"):
        with st.spinner("Explaining the sample..."):
            explain.explain(text, approach, label_i, show_infi)
        

if __name__ == "__main__":
    st.title("Local Explanation")
    run()
