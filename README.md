## Explanation Helper 

- This is a model explanation helper with GUI, which is a revision and an extension of the course project from the seminar Klassifikation und Clustering (WS 2024/25).

- I implemented the GUI with Streamlit for explaining a text classification model locally with different approaches as well as explanation evaluation (for now only `Infidelity`). Some basic EDA and model evaluations with a test dataset given are included as well.

- Training is not a part of this prototype-level app, but it has been done in the original course work in .ipynb format, which is also in the repository. For simplicity, a fine-tuned model will be directly imported from HuggingFace in this demo.

### Problems to resolve
- `feature_selection='none'` in `LimeTextExplainer` does not work as intended. Therefore, I have to implement a LIME for text from scratch. The structure is written in `lime_fix.py`, but there are bugs in it for now (`LIME.attribute` returns `NoneType`).