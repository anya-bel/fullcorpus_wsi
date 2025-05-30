import torch
from tqdm.auto import tqdm


def compute_embeddings(df, tokenizer, model, device, border_column='position', example_column='sentence', verbose=True):
    all_layer_embeddings = dict()
    if verbose:
        iterator = tqdm(df.iterrows(), total=df.shape[0])
    else:
        iterator = df.iterrows()
    for n, row in iterator:
        word_borders = row[border_column]
        tokenized = tokenizer(row[example_column], return_offsets_mapping=True)
        offset = tokenized['offset_mapping'][1:-1]
        if 'deberta' not in model.name_or_path:
            token_idxs = [i for i, token_span in enumerate(offset)
                          if word_borders[0] <= token_span[0] <= word_borders[1]]
        else:
            token_idxs = [i for i, token_span in enumerate(offset)
                          if word_borders[0] - 1 <= token_span[0] <= word_borders[1]]
        if len(token_idxs) == 1:
            token_pos = (token_idxs[0] + 1, token_idxs[0] + 2)
        else:
            token_pos = (token_idxs[0] + 1, token_idxs[-1] + 2)

        tokenized = tokenizer(row[example_column], return_offsets_mapping=False, return_tensors='pt').to(device)
        out = model(**tokenized, output_hidden_states=True)
        for layer in range(len(out['hidden_states'])):
            emb = torch.mean(out['hidden_states'][layer][0][token_pos[0]:token_pos[1]], dim=0).cpu().detach().numpy()
            if layer not in all_layer_embeddings:
                all_layer_embeddings[layer] = []
            all_layer_embeddings[layer].append(emb)

    return all_layer_embeddings
