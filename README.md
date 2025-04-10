## Explanation Helper 

- This is a model explanation helper with GUI, which is a revision and an extension of the course project from the seminar Klassifikation und Clustering (WS 2024/25).

- I implemented the GUI with Streamlit for explaining a text classification model locally with different approaches as well as explanation evaluation (for now only `Infidelity`). Some basic EDA and model evaluations with a test dataset given are included as well.

- Training is not a part of this prototype-level app, but it has been done in [the original coursework](original_coursework.ipynb). For simplicity, a fine-tuned model will be directly imported from HuggingFace in this demo.

### Problems to solve
- `feature_selection='none'` in `LimeTextExplainer` does not work as intended. Therefore, I have to implement a LIME for text from scratch. The structure is written in `lime_fix.py`, but there are bugs in it for now (`LIME.attribute` returns `NoneType`).

## Interface

### Homepage
<img title="Homepage" alt="homepage" src="image\model_loaded.png">

### EDA
<img title="token length distribution" alt="eda1" src="image\eda1.png">
<img title="category distribution" alt="eda1" src="image\eda2.png">

### Classification Evaluation
<img title="classification report" alt="clss1" src="image\clss1.png">
<img title="confusion matrix" alt="clss2" src="image\clss2.png">

### Local Explanation
<img title="SHAP" alt="shap" src="image\shap.png">
<img title="Saliency1" alt="Saliency1" src="image\Saliency1.png">
<img title="Saliency2" alt="Saliency2" src="image\Saliency2.png">

### Find Mismatches
<img title="Find Mismatches" alt="Find Mismatches" src="image\finding_mismatch.png">