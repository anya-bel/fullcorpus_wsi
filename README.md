# SemCorWSI experiments

This repository contains the code for experiments from the paper *"In the LLM era, Word Sense Induction remains unsolved"*.

### Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

### Usage
To run the clustering algorithm, execute `perform_clustering.py` with its arguments:

#### Minimal working example
```
python perform_clustering.py --output_file out
```
#### With optional arguments:
```
python perform_clustering.py --dataset datasets/semcorwsi_dataset_all_dev.csv --algorithm agglomerative --n_clusters_compute silhouette --constraint must-link --output_file out
```

#### Arguments:

* `dataset` : Path to a pandas dataframe containing the data for word sense induction
* `algorithm` : Clustering algorithm to use (agglomerative or x-means)
* `n_clusters_compute ` : Method for determining the number of clusters (silhouette or const+silhouette if must-link is used)
* `constraint ` : Whether to apply must-link or must-link/cannot-link constraints
* `modelname ` : Name of the pre-trained model to use
* `layer ` : The PLM's layer to extract embeddings from
* `device ` : Which device to use
* `add_data_path` : Path to additional unlabeled data for data augmentation
* `output_file ` : Path to save the output results

#### Output
The script generates an `out.pkl` file containing a dictionary with metrics and labels for each example in the dataset.

### Repository content

The **datasets** folder contains:

* *SemCorWSI* dataset
* *Wiktionary* dataset filtered on dev part of SemCorWSI (temporary, as the full version is more than 100MB), 
* *WikiBooks* datasets with 10, 50, 100, 150 examples for dev part of SemCorWSI, 
* *GPT-4o dataset* of generated examples for dev part of SemCorWSI, 
* *Llama 3.1 8B 4bit generated examples dataset* for dev part of SemCorWSI.


The **src** folder contains the main codebase, including the WSIClustering class, which provides core functionality for clustering used in the paper.

