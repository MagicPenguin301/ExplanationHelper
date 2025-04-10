import streamlit as st
import utils
import clss_eval

@utils.after_model_loaded
def run():
    if test_file := st.file_uploader("Upload a test dataset."):
        texts, trues = utils.read_data(test_file)
        clss_eval.evaluate_and_visualize(texts, trues)

if __name__ == "__main__":
    st.title("Classification Evaluation")
    run()