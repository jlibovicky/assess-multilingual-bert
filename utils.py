import os

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sacremoses import MosesTokenizer


def text_data_generator(path, tokenizer, epochs=1, max_len=510):
    for _ in range(epochs):
        with open(path, 'r', encoding='utf-8') as f_txt:
            for line in f_txt:
                sentence = line.strip()
                yield tokenizer.encode(sentence, max_length=max_len)


def batch_generator(generator, size, tokenizer, padding=True):
    """Take data generator and return batches of given size."""
    items = []

    for item in generator:
        items.append(item)

        if len(items) >= size:
            if padding:
                yield pad_sentences(items, tokenizer)
            else:
                yield torch.stack(items)
            items = []
    if items:
        if padding:
            yield pad_sentences(items, tokenizer)
        else:
            yield torch.stack(items)


def pad_sentences(sentences, tokenizer):
    max_len = max(len(ex) for ex in sentences)
    padded_batch = torch.zeros(
        len(sentences), max_len, dtype=torch.int64) + tokenizer.pad_token_id
    for i, ex in enumerate(sentences):
        for j, idx in enumerate(ex):
            padded_batch[i, j] = idx
    return padded_batch


def get_repr_from_layer(model, data, layer, pad_index, mean_pool=False):
    mask = (data != pad_index).float()
    if layer >= 0:
        layer_output = model(data, attention_mask=mask)[-1][layer]
        if mean_pool:
            mask = mask.unsqueeze(2)
            lengths = mask.long().sum(1)

            # Mask out [CLS] and [SEP] symbols as well.
            # mask[:, lengths - 1] = 0
            # mask[:, 0] = 0
            return (layer_output * mask).sum(1) / mask.sum(1)

        # Otherwise just take [CLS]
        return layer_output[:, 0]

    if layer == -1:
        model_output = model(data, attention_mask=mask)[0]
        if len(model_output) == 3:
            if mean_pool:
                raise ValueError(f"Cannot mean-pool the default vector.")
            return model_output[1]
        assert mean_pool
        mask = mask.unsqueeze(2)
        return (model_output[0] * mask).sum(1) / mask.sum(1)

    raise ValueError(f"Invalid layer {layer}.")


def vectors_for_sentence(
        tokenizer, model, sentence, layer, skip_tokenization=False):
    if skip_tokenization:
        tokens = sentence
    else:
        tokens = tokenizer.tokenize(sentence)

    tokenized = [tokenizer.cls_token] + tokens[:510] + [tokenizer.sep_token]
    token_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(tokenized)).unsqueeze(0)

    layer_output = model(token_ids)[-1][layer]

    return layer_output.squeeze(0)[1:-1], tokenized[1:-1]


PRETRAINED_BERTS = set([
    "bert-base-uncased", "bert-large-uncased", "bert-base-cased",
    "bert-base-multilingual-cased", "bert-base-multilingual-uncased",
    "bert-base-chinese", 'xlm-roberta-base', 'distilbert-base-multilingual-cased'])


def load_bert(bert_spec, device):
    """Load pretrained BERT, either standard or from a file."""

    if not os.path.isdir(bert_spec) and bert_spec not in PRETRAINED_BERTS:
        raise ValueError(
            f"{bert_spec} is not a directory neither a pretrained BERT id."
            f"Available: {' '.join(PRETRAINED_BERTS)}")

    tokenizer = AutoTokenizer.from_pretrained(
        bert_spec, do_lower_case=bert_spec.endswith("-uncased"))
    model = AutoModel.from_pretrained(bert_spec, output_hidden_states=True).to(device)
    model.eval()

    model_dim = None
    if hasattr(model.config, 'dim'):
        model_dim = model.config.dim
    if hasattr(model.config, 'hidden_size'):
        model_dim = model.config.hidden_size
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


def load_word_embeddings(path):
    if os.path.exists(path + ".bin"):
        return joblib.load(path + ".bin")

    embeddings_dic = {}
    with open(path) as f_vec:
        count_str, dim_str = f_vec.readline().strip().split()
        _, dim = int(count_str), int(dim_str)

        for line in f_vec:
            word, vec_str = line.strip().split(maxsplit=1)
            vector = np.fromstring(vec_str, sep=" ", dtype=np.float)
            if vector.shape == (dim,):
                embeddings_dic[word] = vector

    return embeddings_dic


TOKENIZERS = {}


def get_tokenizer(lng):
    if lng not in TOKENIZERS:
        TOKENIZERS[lng] = MosesTokenizer(lng)
    return TOKENIZERS[lng]


def mean_word_embedding(embeddings, sentence, lng, mean_pool=True,
                        skip_tokenization=False):
    unk = embeddings["</s>"]
    if skip_tokenization:
        tokens = sentence.split(" ")
    else:
        tokens = get_tokenizer(lng).tokenize(sentence)
    embedded_tokens = [embeddings.get(tok.lower(), unk) for tok in tokens]
    if mean_pool:
        return np.mean(embedded_tokens, axis=0)
    return np.stack(embedded_tokens), tokens


def word_embeddings_for_file(path, embeddings, lng, mean_pool=True,
                             skip_tokenization=False):
    embedded_sentences = []
    with open(path) as f_txt:
        for line in f_txt:
            embedded_sentences.append(
                mean_word_embedding(embeddings, line.strip(), lng,
                                    mean_pool=mean_pool,
                                    skip_tokenization=skip_tokenization))
    return embedded_sentences
