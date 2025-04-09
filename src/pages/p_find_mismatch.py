import streamlit as st, pandas as pd
import utils, find_mismatch

@utils.after_model_loaded
def run():
    if test_file := st.file_uploader("Upload your test dataset here."):
        try:
            with st.spinner("Reading the file..."):
                texts, labels = utils.read_data(test_file)
            with st.spinner("Finding mismatches..."):
                # triples: (text, true, pred)
                mismatches = find_mismatch.find_mismatches(texts, labels)
        except Exception as e:
            st.error("An error has occurred during finding mismatches: {e}")
        mis_df = pd.DataFrame(mismatches, columns=["text", "true", "pred"])
        st.download_button(
                label="Download mismatches",
                data=mis_df.to_csv().encode("utf-8"),
                file_name="mismatches.csv",
                mime="text/csv",
                icon=":material/download:",
            )


if __name__ == "__main__":
    st.title("Find Wrong Predictions")
    with st.expander("Why do I need this?"):
        st.write("During explanation, it's meaningful to understand where and why the model made mistakes. " \
        "This page can help us find mismatches quickly. Use samples in the exported file for detailed *Local Explanation*.")
    run()