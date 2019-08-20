import torch

def data_generator(path, tokenizer):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()

            # 512 is the maximum input size of BERT
            tokens = tokenizer.tokenize(sentence)
            tokenized = ["[CLS]"] + tokens[:510] + ["[SEP]"]
            token_ids = tokenizer.convert_tokens_to_ids(tokenized)
            yield torch.tensor(token_ids)


def batch_generator(generator, size):
    sentences = []

    for sentence in generator:
        sentences.append(sentence)

        if len(sentences) >= size:
            yield pad_sentences(sentences)
            sentences = []
    if sentences:
        yield pad_sentences(sentences)


def pad_sentences(sentences):
    max_len = max(ex.size(0) for ex in sentences)
    padded_batch = torch.zeros(len(sentences), max_len, dtype=torch.int64)
    for i, ex in enumerate(sentences):
        padded_batch[i,:ex.size(0)] = ex
    return padded_batch


def get_repr_from_layer(model, data, layer, mean_pool=False):
    if layer >= 0:
        layer_output = model(data, torch.zeros_like(data))[0][layer]
        if mean_pool:
            mask = (data != 0).float().unsqueeze(2)
            lengths = mask.long().sum(1)

            # Mask out [CLS] and [SEP] symbols as well.
            mask[:,lengths - 1] = 0
            mask[:,0] = 0
            return (layer_output * mask).sum(1) / mask.sum(1)
        else:
            return layer_output[:, 0]
    elif layer == -1:
        if mean_pool:
            raise ValueError(f"Cannot mean-pool the default vector.")
        return model(data, torch.zeros_like(data))[1]
    else:
        raise ValueError(f"Invalid layer {layer}.")


def vectors_for_sentence(tokenizer, model, data, layer, mean_pool):
    tokens = tokenizer.tokenize(sentence)
    tokenized = ["[CLS]"] + tokens[:510] + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokenized)

    layer_output = model(
        token_ids.unsqueeze(0), torch.zeros_like(data))[0][layer]
