from captum.attr import LimeBase
from captum._utils.models.linear_model import SkLearnLasso
from transformers import BertTokenizer, BertForSequenceClassification
import torch, utils
from torch.nn import functional as F


def bernoulli_perturb(text_tensor, prob=0.5):
    """
    Randomly masks tokens using a Bernoulli distribution.
    Returns a perturbed input tensor.
    """
    mask = torch.bernoulli(torch.ones_like(text_tensor.float()) * prob).long()
    mask[0] = 1  # Ensuring [CLS] token is always retained
    return text_tensor * mask.float()


# Define the predict function for LIME
def predict(encoding):
    # Convert the encoding to the model input format
    mask = (encoding > 0).long()
    output = model(encoding.long(), mask)
    return output.logits


def cosine_similarity_func(original_inp, perturbed_inp, _, **kwargs):
    # Compute cosine similarity between the original and perturbed inputs
    return 1 - F.cosine_similarity(original_inp, perturbed_inp, dim=1)

def interp_to_input(interp_sample, original_input, **kwargs):
    # interp_sample is a binary mask for the perturbed tokens
    # original_input is the tokenized representation of the text
    return original_input.float() * interp_sample.float()

# Initialize the model and tokenizer
tokenizer = utils.tokenizer
model = utils.model

# Custom LIME setup using SkLearnLasso as the surrogate model
LIME = LimeBase(
    predict,
    interpretable_model=SkLearnLasso(alpha=0.08),
    perturb_func=bernoulli_perturb,
    perturb_interpretable_space=True,
    from_interp_rep_transform=interp_to_input,
    to_interp_rep_transform=None,
    similarity_func=cosine_similarity_func,
)


def get_lime_attributions(text, label_i):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    return (
        LIME.attribute(
            torch.tensor(input_ids).unsqueeze(0), target=label_i, n_samples=10
        ),
        input_ids,
    )
