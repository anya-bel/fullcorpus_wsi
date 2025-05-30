from difflib import SequenceMatcher

import spacy
from tqdm.auto import tqdm

nlp = spacy.load("en_core_web_lg", disable=['ner'])


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_target_borders(lemma, example, similarity_threshold=0.5, spacy_model='en_core_web_lg'):
    """Finds borders of the target word in the example
    Parameters
    -------
    lemma : str
        A target word to be found in the example
    example : str
        An example where a target word should be found
    similarity_threshold : float, optional
        A threshold below which None is returned
    spacy_model : name of the spacy model to use (usually either en_core_web_lg for english or fr_dep_news_trf)
    Returns
    -------
    tuple or None
        Two integers which are a starting and an ending position of a lemma in the example

    """
    nlp = spacy.load(spacy_model, disable=['ner'])
    sent = nlp(example)
    best_sim, best_position = 0, 0
    verb_position = None
    for elem in sent:
        if elem.lemma_ == lemma or elem.text == lemma:
            verb_position = (elem.idx, elem.idx + len(elem.text))
            break

        sim_lemma = similar(str(lemma), str(elem.lemma_))
        sim_form = similar(str(lemma), str(elem.text))
        if sim_lemma > best_sim or sim_form > best_sim:
            if sim_lemma > sim_form:
                best_sim = sim_lemma
            else:
                best_sim = sim_form
            best_position = elem.i
            if best_sim >= similarity_threshold:
                verb_position = (elem.idx, elem.idx + len(elem.text))
    else:
        if best_sim < similarity_threshold:
            print(
                f'For the target {lemma}, the required match is not found, the closest one is {sent[best_position]} (similarity {best_sim})')
            return None

    return verb_position


def add_target_borders(df, similarity_threshold=0.5, force_border=False, spacy_model='en_core_web_lg',
                       example_col='Example', target_col='Target', border_col='Border'):
    """Adds borders of the target word into the dataframe
    Parameters
    -------
    df : pandas.DataFrame
        a DataFrame containing at least two columns: example_col and target_col. target_col should contain a word to look for in the example_col
    similarity_threshold : float, optional
        A threshold below which the border value is not filled in as the target word is not found in the example
    force_border : whether to add the found border even when candidate's similarity is below the threshold or leave it as None
    spacy_model : name of the spacy model to use (usually either en_core_web_lg for english or fr_dep_news_trf)
    example_col : name of the column where the examples are stored
    border_col : name of the column where the found borders should be stored
    target_col : name of the column where the lemmas are stored
    Returns
    -------
    pandas.DataFrame
        new DataFrame with the column Border (border_col) where a tuple with the starting and ending position
        of a target word is given. If the word is not found in an example, the Border value is None

    """
    nlp = spacy.load(spacy_model, disable=['ner'])
    df_with_borders = df.copy()
    df_with_borders[border_col] = None

    for n, row in tqdm(df_with_borders.iterrows(), total=df_with_borders.shape[0]):
        sent = nlp(row[example_col])
        best_sim, best_position = 0, 0
        verb_position = None
        forced_position = None
        for elem in sent:
            if elem.lemma_ == row[target_col] or elem.text == row[target_col]:
                verb_position = (elem.idx, elem.idx + len(elem.text))
                break

            dist_lemma = similar(str(row[target_col]), str(elem.lemma_))
            dist_form = similar(str(row[target_col]), str(elem.text))
            if dist_lemma > best_sim or dist_form > best_sim:
                if dist_lemma > dist_form:
                    best_sim = dist_lemma
                else:
                    best_sim = dist_form
                best_position = elem.i
                if best_sim >= similarity_threshold:
                    verb_position = (elem.idx, elem.idx + len(elem.text))
                else:
                    forced_position = (elem.idx, elem.idx + len(elem.text))
        else:
            if best_sim < similarity_threshold:
                print(
                    f'Sent {n}. For the target {row[target_col]}, the best match is not found, the closest one is {sent[best_position]} (similarity {best_sim})')
                if force_border:
                    verb_position = forced_position

        df_with_borders.at[n, border_col] = verb_position

    return df_with_borders
