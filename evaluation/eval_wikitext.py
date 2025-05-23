import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_from_disk


def evaluate_perplexity(model, tokenizer,dev):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_from_disk("/root/autodl-tmp/datasets/wikitext")["test"]
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids

    seqlen = 1024
    model.to(dev)
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index]
            with torch.no_grad():
                logits = model(batch.to(dev)).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:].to(dev)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()