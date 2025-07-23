---
layout: default
title: SemCor-WSI
---

## Framework for full-corpus word sense induction evaluation

This website contains the **SemCor-WSI** dataset designed to evaluate word sense induction (WSI) systems in full-corpus settings. This framework was proposed in the ACL Findings paper ["In the LLM era, Word Sense Induction remains unsolved"](https://aclanthology.org/2025.findings-acl.882.pdf).

The dataset is derived from the **SemCor 3.0** corpus, which is the property of Princeton University. It is distrubted under the SemCor's original license, included within the SemCor package, available at:  [http://web.eecs.umich.edu/~mihalcea/ downloads/semcor/semcor3.0.tar.gz.](http://web.eecs.umich.edu/~mihalcea/ downloads/semcor/semcor3.0.tar.gz.)

To evaluate your system's output, we recommend using the **B-Cubed metric** available via the [bcubed-metrics Python package](https://pypi.org/project/bcubed/). For details on this choice of evaluation metric, please refer to the [paper](https://aclanthology.org/2025.findings-acl.882.pdf).

---

## Download the Dataset

#### - [Download Full Dataset (ZIP)](/files/dataset.zip)
#### - [Readme & Format Description](/files/README.txt)


The package contains separate files for three parts of speech: adjectives, nouns and verbs, along with a full version containing all POS examples. Dataset statistics are given in the table below. For more details about the dataset composition, please refer to the [paper](https://aclanthology.org/2025.findings-acl.882.pdf).

| POS  | Dev Instances | Dev Lemmas | Dev Polysemy | Test Instances | Test Lemmas | Test Polysemy | Full SemCor Polysemy |
|------|-----------|----------|--------------|------------|-----------|----------------|------------------|
| Adjective  | 4909      | 433      | 1.69 (±1.3)   | 4772       | 427       | 1.69 (±1.2)     | 1.64 (±1.2)       |
| Noun | 5394      | 479      | 1.75 (±1.4)   | 5694       | 493       | 1.73 (±1.4)     | 1.71 (±1.4)       |
| Verb | 5005      | 359      | 2.39 (±2.2)   | 4979       | 367       | 2.36 (±2.4)     | 2.34 (±2.5)       |
| All  | 15308     | 1271     | 1.94 (±1.7)   | 15445      | 1287      | 1.91 (±1.7)     | 2.10 (±2.2)       |


## Citation

If you use this dataset, please cite the following paper:

> Anna Mosolova, Marie Candito, and Carlos Ramisch. 2025. [In the LLM era, Word Sense Induction remains unsolved.](https://aclanthology.org/2025.findings-acl.882.pdf) In Findings of the Association for Computational Linguistics: ACL 2025, pages 17161–17178, Vienna, Austria. Association for Computational Linguistics.
