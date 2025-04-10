from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import transformers, streamlit as st
import matplotlib.pyplot as plt

import numpy as np
import utils
import pandas as pd


def evaluate_and_visualize(texts, true_labels):
    # sourcery skip: extract-duplicate-method, extract-method
    classifier = utils.classifier

    with st.spinner("Getting predictions..."):
        predicted_labels = []
        for text in texts:
            predicted_label = classifier(text)[0]["label"]
            predicted_labels.append(predicted_label)

    with st.spinner("Creating the classification report..."):
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.write("**Classification Report:**")
        st.dataframe(df_report)

    with st.spinner("Generating the confusion matrix..."):
        unique_labels = np.unique(true_labels)
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

        # Use the same ordered labels for display
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=unique_labels
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.title("Confusion Matrix")
        st.pyplot(fig)

    # ROC curve (only for binary classification)
    with st.spinner("Generating ROC curve..."):
        if len(np.unique(true_labels)) == 2:
            try:
                label_mapping = {
                    label: i for i, label in enumerate(np.unique(true_labels))
                }
                numerical_true = [label_mapping[label] for label in true_labels]
                numerical_pred_probs = [
                    classifier(text)[0][1]["score"] for text in texts
                ]  # Assuming probability for positive class is at index 1

                fpr, tpr, _ = roc_curve(numerical_true, numerical_pred_probs)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], "k--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

            except Exception as e:
                st.error(f"An error occurred during ROC curve calculation: {e}")
