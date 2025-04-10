import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import transformers
import streamlit as st
from captum.metrics import infidelity

classifier = transformers.pipeline(
    "text-classification",
    model=utils.model,
    tokenizer=utils.tokenizer,
    device=utils.device,
    return_all_scores=True,
)


def perturb_input(text, num_samples=20):
    words = utils.tokenizer.tokenize(text)
    perturbed_texts = []
    deltas = []

    for _ in range(num_samples):
        mask = np.random.binomial(1, 0.5, size=len(words))
        new_words = [w if keep else "[MASK]" for w, keep in zip(words, mask)]
        perturbed_text = " ".join(new_words)
        perturbed_texts.append(perturbed_text)
        deltas.append(mask.astype(float))

    return perturbed_texts, deltas


class EvalForSHAP:
    @staticmethod
    def predict(texts):
        outputs = classifier(texts)
        return np.array([[entry["score"] for entry in result] for result in outputs])

    @staticmethod
    def compute_infidelity(text, shap_values, num_samples=None):
        if not num_samples:
            try:
                num_samples = int(
                    utils.config["expl_eval"]["infidelity"]["num_samples"]
                )
            except Exception as e:
                st.warning(f"Failed to read config: {e}")
                num_samples = 10

        base_pred = EvalForSHAP.predict([text])[0]
        attribution = shap_values.values[0][
            : len(utils.tokenizer.tokenize(text))
        ]  # shap gives per-token values

        perturbed_texts, deltas = perturb_input(text, num_samples)
        perturbed_preds = EvalForSHAP.predict(perturbed_texts)

        infidelities = []
        for delta_vec, pert_pred in zip(deltas, perturbed_preds):
            delta_pred = base_pred - pert_pred
            attribution_sum = np.dot(delta_vec, attribution)
            infidelity = np.square(attribution_sum - np.sum(delta_pred))
            infidelities.append(infidelity)

        return np.mean(infidelities)


class EvalForLIME:
    @staticmethod
    def predict(texts):
        outputs = classifier(texts)
        return np.array([[entry["score"] for entry in result] for result in outputs])

    @staticmethod
    def compute_infidelity(text, scores, desired_label_i, num_samples=None):
        # print(type(scores))
        if not num_samples:
            try:
                num_samples = int(
                    utils.config["expl_eval"]["infidelity"]["num_samples"]
                )
            except Exception as e:
                st.warning(f"Failed to read config: {e}")
                num_samples = 10

        # Get model predictions for the original and perturbed texts
        base_pred = EvalForLIME.predict([text])[0][desired_label_i]
        perturbed_texts, deltas = perturb_input(text, num_samples)
        perturbed_preds = EvalForLIME.predict(perturbed_texts)

        infidelities = []
        for delta_vec, pert_pred in zip(deltas, perturbed_preds):
            delta_pred = base_pred - pert_pred[desired_label_i]
            attribution_sum = np.dot(delta_vec, scores)
            infidelity = np.square(attribution_sum - delta_pred)
            infidelities.append(infidelity)

        return np.mean(infidelities)


class EvalForSaliency:
    @staticmethod
    def predict(texts):
        outputs = classifier(texts)
        # st.text(outputs)
        return np.array([[entry["score"] for entry in result] for result in outputs])

    @staticmethod
    def compute_infidelity(text, attributions, label_i, num_samples=None):
        if not num_samples:
            try:
                num_samples = int(
                    utils.config["expl_eval"]["infidelity"]["num_samples"]
                )
            except Exception as e:
                st.warning(f"Failed to read config: {e}")
                num_samples = 10

        base_pred = EvalForSaliency.predict([text])[0][label_i]

        # Truncate attributions to match number of tokens
        token_count = len(utils.tokenizer.tokenize(text))
        attributions = attributions[:token_count]

        perturbed_texts, deltas = perturb_input(text, num_samples)
        perturbed_preds = EvalForSaliency.predict(perturbed_texts)

        infidelities = []
        for delta_vec, pert_pred in zip(deltas, perturbed_preds):
            delta_pred = base_pred - pert_pred[label_i]
            attribution_sum = np.dot(delta_vec[: len(attributions)], attributions)
            infidelity = np.square(attribution_sum - delta_pred)
            infidelities.append(infidelity)

        return np.mean(infidelities)
