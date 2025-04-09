import transformers, utils

def find_mismatches(texts, labels):
    classifier = transformers.pipeline(
        "text-classification",
        model=utils.model,
        tokenizer=utils.tokenizer,
        device=utils.device,
        return_all_scores=False
    )

    predictions = classifier(texts)

    # Get predicted labels (as indices or strings depending on your label format)
    pred_labels = [pred['label'] for pred in predictions]

    # Return mismatches: (text, true_label, predicted_label)
    return list(filter(
        lambda tpl: tpl[1] != tpl[2],
        zip(texts, labels, pred_labels)
    ))
