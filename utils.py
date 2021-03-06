import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


def text_data_generator(path, tokenizer, epochs=1, max_len=510):
    for _ in range(epochs):
        with open(path, 'r', encoding='utf-8') as f_txt:
            for line in f_txt:
                sentence = line.strip()

                # 512 is the maximum input size of BERT
                tokens = tokenizer.tokenize(sentence)
                tokenized = ["[CLS]"] + tokens[:max_len] + ["[SEP]"]
                token_ids = tokenizer.convert_tokens_to_ids(tokenized)
                yield torch.tensor(token_ids)


def batch_generator(generator, size, padding=True):
    """Take data generator and return batches of given size."""
    items = []

    for item in generator:
        items.append(item)

        if len(items) >= size:
            if padding:
                yield pad_sentences(items)
            else:
                yield torch.stack(items)
            items = []
    if items:
        if padding:
            yield pad_sentences(items)
        else:
            yield torch.stack(items)


def pad_sentences(sentences):
    max_len = max(ex.size(0) for ex in sentences)
    padded_batch = torch.zeros(len(sentences), max_len, dtype=torch.int64)
    for i, ex in enumerate(sentences):
        padded_batch[i, :ex.size(0)] = ex
    return padded_batch


def get_repr_from_layer(model, data, layer, mean_pool=False):
    mask = (data != 0).float()
    if layer >= 0:
        layer_output = model(data, attention_mask=mask)[0][layer]
        if mean_pool:
            mask = mask.unsqueeze(2)
            lengths = mask.long().sum(1)

            # Mask out [CLS] and [SEP] symbols as well.
            mask[:, lengths - 1] = 0
            mask[:, 0] = 0
            return (layer_output * mask).sum(1) / mask.sum(1)

        # Otherwise just take [CLS]
        return layer_output[:, 0]

    if layer == -1:
        if mean_pool:
            raise ValueError(f"Cannot mean-pool the default vector.")
        return model(data, attention_mask=mask)[1]

    raise ValueError(f"Invalid layer {layer}.")


def vectors_for_sentence(
        tokenizer, model, sentence, layer, skip_tokenization=False):
    if skip_tokenization:
        tokens = sentence
    else:
        tokens = tokenizer.tokenize(sentence)

    tokenized = ["[CLS]"] + tokens[:510] + ["[SEP]"]
    token_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokenized)).unsqueeze(0)

    layer_output = model(
        token_ids, torch.zeros_like(token_ids))[0][layer]

    return layer_output.squeeze(0)[1:-1], tokenized[1:-1]


PRETRAINED_BERTS = set([
    "bert-base-uncased", "bert-large-uncased", "bert-base-cased",
    "bert-base-multilingual-cased", "bert-base-multilingual-uncased",
    "bert-base-chinese"])


def load_bert(bert_spec, device):
    """Load pretrained BERT, either standard or from a file."""

    if not os.path.isdir(bert_spec) and bert_spec not in PRETRAINED_BERTS:
        raise ValueError(
            f"{bert_spec} is not a directory neither a pretrained BERT id."
            f"Available: {' '.join(PRETRAINED_BERTS)}")

    tokenizer = BertTokenizer.from_pretrained(
        bert_spec, do_lower_case=bert_spec.endswith("-uncased"))
    model = BertModel.from_pretrained(bert_spec).to(device)
    model.eval()

    model_dim = model.encoder.layer[-1].output.dense.out_features
    vocab_size = model.embeddings.word_embeddings.weight.size(0)

    return tokenizer, model, model_dim, vocab_size


def get_lng_database():
    lng_info = {}
    with open("bert_languages_complete.tsv") as f_lng:
        for line in f_lng:
            fields = line.strip().split("\t")

            record = {
                "name": fields[0],
                "iso": fields[2]}

            if len(fields) > 3 and fields[3]:
                record["genus"] = fields[3]
            if len(fields) > 4 and fields[4]:
                record["family"] = fields[4]
            if len(fields) > 5 and fields[5]:
                record["svo"] = fields[5]
            if len(fields) > 6 and fields[6]:
                record["sv"] = fields[6]
            if len(fields) > 7 and fields[7]:
                record["vo"] = fields[7]
            if len(fields) > 8 and fields[8]:
                record["adj-noun"] = fields[8]

            lng_info[fields[0]] = record
    return lng_info
