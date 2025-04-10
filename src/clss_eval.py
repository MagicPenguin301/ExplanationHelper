from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
)
import transformers, streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils


def evaluate_and_visualize(texts, true_labels):
    # sourcery skip: extract-duplicate-method, extract-method

    # Create a prediction pipeline
    classifier = utils.classifier

    with st.spinner("Getting predictions..."):
        predicted_labels = []
        for text in texts:
            predictions = classifier(text)
            predicted_label = max(predictions[0], key=lambda x: x["score"])["label"]
            predicted_labels.append(predicted_label)

    with st.spinner("Creating the classification report..."):
        report = classification_report(true_labels, predicted_labels)
        st.text("**Classification Report:**")
        st.text(report)

    with st.spinner("Generating the confusion matrix..."):
        cm = confusion_matrix(true_labels, predicted_labels)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax_cm)
        ax_cm.set_xlabel("Predicted Labels")
        ax_cm.set_ylabel("True Labels")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

    # ROC curve (only for binary classification)
    with st.spinner("Generating ROC curve..."):
        if len(np.unique(true_labels)) == 2:
            try:
                # Convert labels to numerical format if needed
                label_mapping = {label: i for i, label in enumerate(np.unique(true_labels))}
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
