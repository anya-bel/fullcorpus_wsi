SemCor-WSI Dataset
------------------

This dataset accompanies the paper:

  In the LLM Era, Word Sense Induction Remains Unsolved  
  Anna Mosolova, Marie Candito, and Carlos Ramisch  
  Findings of ACL 2025  
  https://aclanthology.org/2025.findings-acl.882.pdf


Description
-----------

SemCor-WSI is a dataset for full-corpus word sense induction (WSI) evaluation.
It is extracted from the SemCor 3.0 corpus and provides disambiguated instances for three parts of speech: adjectives, nouns, and verbs. The dataset is split into development and test sets. A combined version ("all") includes all POS together.

Each instance corresponds to a single word occurrence in context and includes its disambiguated WordNet sense.

Data Format
-----------

Each file is a comma-separated file. Each line corresponds to one instance with the following fields:

  sentence   : Full sentence containing the target word
  lemma      : Lemmatized version of the target word
  lexsn      : WordNet lexical sense key (e.g., 5:00:00:real:00)
  wnsn       : WordNet sense number (e.g., 2)
  pos        : Part of speech tag (JJ, NN, VB)
  position   : first and last letter positions of the target word in the sentence

Example line:

  sentence: Historical existence is a created good .
  lemma   : historical	5:00:00:real:00  
  lexsn   : 5:00:00:real:00  
  wnsn    : 2
  pos     : JJ  
  position: (0, 10)

Files
-----

  dev_adj.csv     - Development set, adjectives
  dev_noun.csv    - Development set, nouns
  dev_verb.csv    - Development set, verbs
  dev_all.csv     - Development set, all POS
  test_adj.csv    - Test set, adjectives
  test_noun.csv   - Test set, nouns
  test_verb.csv   - Test set, verbs
  test_all.csv    - Test set, all POS

License
-------

This dataset is derived from SemCor 3.0, which is the property of Princeton University.
It is redistributed under the original SemCor license, available at:

  http://web.eecs.umich.edu/~mihalcea/downloads/semcor/semcor3.0.tar.gz

Citation
--------

If you use this dataset, please cite the following paper:

  Anna Mosolova, Marie Candito, and Carlos Ramisch. 2025.  
  In the LLM Era, Word Sense Induction Remains Unsolved.  
  In Findings of the Association for Computational Linguistics: ACL 2025.  
  https://aclanthology.org/2025.findings-acl.882
