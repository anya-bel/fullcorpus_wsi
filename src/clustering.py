import sys

sys.path.insert(0, '..')

AG_linkage = 'average'
AG_affinity = 'euclidean'
SILH_max_cl = 15
XMEANS_tolerance = 0.001
XMEANS_min_cl = 1
XMEANS_max_cl = 15

from statistics import mean

import numpy as np
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm
import pandas as pd

from src.evaluate import evaluate_clusters


class WSIClustering:
    def __init__(self, algorithm='agglomerative', n_cluster_compute='silhouette', constraint=False,
                 constraint_sense_col='Definition', target_col=('lemma', 'pos'),
                 sense_col='lexsn', compute_metrics=True):
        """
        :param algorithm: agglomerative or x-means
        :param n_cluster_compute: silh or const+silhouette (not applicable for X-means)
        :param constraint: must-link/cannot-link or must-link
        :param constraint_sense_col: sense column to use in the constraint_dataframe (default: Definition)
        :param target_col: columns which will be used for per lemma clustering (default: ('lemma', 'pos'))
        :param sense_col: sense column to use in original dataframe (default: lexsn)
        :param compute_metrics: True returns all metrics computed using sense_col as gold, otherwise only cluster labels are returned
        """
        self.algorithm = algorithm
        self.n_cluster_compute = n_cluster_compute
        self.constraint = constraint
        self.constraint_sense_col = constraint_sense_col
        self.target_col = list(target_col)
        self.sense_col = sense_col
        self.compute_metrics = compute_metrics
        if self.constraint not in ['must-link', 'must-link/cannot-link', False]:
            raise ValueError(f'Constraint "{self.constraint}" is not available.')
        if self.algorithm not in ['agglomerative', 'x-means']:
            raise ValueError(
                f'algorithm "{self.algorithm}" is not implemented. available options are: agglomerative, x-means')
        if not self.constraint and self.n_cluster_compute == 'const+silhouette':
            raise ValueError('Constraints gold number of clusters is not available with constraint=False')

    def fit(self, dataframe, embeddings, additional_data=False, additional_embeddings=False, constraint_dataframe=False,
            constraint_embeddings=False):
        """
        :param dataframe: the pandas dataframe with occurrences containing sense_col and target_col
        :param embeddings: numpy array containing a vector for each row in dataframe
        :param additional_data: pandas dataframe with new examples
        :param additional_embeddings: numpy array containing a vector for each row in additional_data
        :param constraint_dataframe: pandas dataframe with sense-annotated examples
        :param constraint_embeddings:numpy array containing a vector for each row in constraint_dataframe
        :return: dict either with metrics and labels or with labels only (depends on compute_metrics parameter)
        """
        embeddings = np.array(embeddings)
        self.additional_data = additional_data

        if dataframe.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f'dataframe and embeddings have different number of instances ({dataframe.shape[0]} and {embeddings.shape[0]})')

        size_per_lemma = {}
        all_results = {}
        all_labels = np.zeros((dataframe.shape[0]), dtype=object)

        for groupname, groupdf in tqdm(dataframe.groupby(self.target_col)):
            size_per_lemma[groupname] = groupdf.shape[0]
            lemma_vectors = embeddings[groupdf.index]

            if isinstance(self.additional_data, pd.DataFrame):
                self.all_additional_lemmas = [x[0] for x in additional_data.groupby(self.target_col)]
                self.additional_vectors = self.add_data(groupname, additional_data, additional_embeddings)

            if self.constraint:
                self.all_constraint_lemmas = [x[0] for x in constraint_dataframe.groupby(self.target_col)]
                self.constraint_lemma_df, self.constraint_lemma_vectors = self.add_constraint_data(groupname,
                                                                                                   constraint_dataframe,
                                                                                                   constraint_embeddings)

            final_labels = self.cluster(groupdf, lemma_vectors)

            if self.compute_metrics:
                all_results[groupname] = evaluate_clusters(groupdf[self.sense_col].to_list(), final_labels)
                all_results[groupname].update({'mean n clusters': len(set(final_labels))})
            all_labels[groupdf.index] = [f'{groupname}_{x}' for x in final_labels]

        if self.compute_metrics:
            total = sum(size_per_lemma.values())
            lemma_for_metrics = list(size_per_lemma.keys())[0]
            output_results = {metric: mean([all_results[lemma][metric] for lemma in all_results]) for metric in
                              all_results[lemma_for_metrics]}
            results_w_mean = {
                metric: sum([all_results[lemma][metric] * size_per_lemma[lemma] / total for lemma in all_results])
                for metric in all_results[lemma_for_metrics]}
            output_results.update({f'{k}_weighted': v for k, v in results_w_mean.items()})
            output_results.update({'labels': list(all_labels)})
        else:
            output_results = {'labels': list(all_labels)}

        return output_results

    def cluster(self, groupdf, vectors):
        if self.algorithm == 'agglomerative':
            if self.constraint:
                if self.constraint_lemma_df.shape[0] > 0:
                    constraint_n_senses = self.constraint_lemma_df[self.constraint_sense_col].unique().shape[0]
                    constraint_df_shape = self.constraint_lemma_df.shape[0]

                    if isinstance(self.additional_data, pd.DataFrame) and self.additional_vectors.shape[0] > 0:
                        all_dists = euclidean_distances(
                            np.concatenate([self.constraint_lemma_vectors, vectors, self.additional_vectors]))
                        vectors = np.concatenate([self.constraint_lemma_vectors, vectors, self.additional_vectors])
                    else:
                        all_dists = euclidean_distances(np.concatenate([self.constraint_lemma_vectors, vectors]))
                        vectors = np.concatenate([self.constraint_lemma_vectors, vectors])
                    defs = self.constraint_lemma_df[self.constraint_sense_col].to_numpy()
                    defs_shape = defs.shape[0]
                    for i in range(defs_shape):
                        for j in range(defs_shape):
                            if defs[i] == defs[j]:
                                all_dists[i, j] = 0
                            if self.constraint == 'must-link/cannot-link':
                                all_dists[i, j] = 10000000000

                    if self.n_cluster_compute == 'silhouette':
                        n_clusters = self.compute_silhouette_clusters(vectors.shape[0], vectors=vectors)
                        tmp_clustering = AgglomerativeClustering(linkage=AG_linkage, n_clusters=n_clusters,
                                                                 metric='precomputed')
                    elif self.n_cluster_compute == 'const+silhouette':
                        tmp_clustering = AgglomerativeClustering(linkage=AG_linkage, n_clusters=constraint_n_senses,
                                                                 metric='precomputed')
                    tmp_clustering.fit(all_dists)
                    tmp_labels = tmp_clustering.labels_

                    if isinstance(self.additional_data, pd.DataFrame) and self.additional_vectors.shape[0] > 0:
                        add_shape = self.additional_vectors.shape[0]
                        final_labels = tmp_labels[constraint_df_shape:-add_shape]
                    else:
                        final_labels = tmp_labels[constraint_df_shape:]
                else:
                    final_labels = self.agglomertive_simple(groupdf, vectors)
            else:
                if isinstance(self.additional_data, pd.DataFrame) and self.additional_vectors.shape[0] > 0:
                    vectors = np.concatenate([vectors, self.additional_vectors])
                    tmp_labels = self.agglomertive_simple(groupdf, vectors)
                    add_shape = self.additional_vectors.shape[0]
                    final_labels = tmp_labels[:-add_shape]
                else:
                    final_labels = self.agglomertive_simple(groupdf, vectors)

        elif self.algorithm == 'x-means':
            centers = kmeans_plusplus_initializer(vectors, XMEANS_min_cl).initialize()
            xmeans_instance = xmeans(vectors, centers, XMEANS_max_cl, tolerance=XMEANS_tolerance,
                                     splitting_type=splitting_type.BAYESIAN_INFORMATION_CRITERION)
            xmeans_instance.process()
            final_labels = xmeans_instance.predict(vectors)

        return final_labels

    def agglomertive_simple(self, groupdf, vectors):
        if vectors.shape[0] > 1:
            n_clusters = self.compute_silhouette_clusters(groupdf.shape[0], vectors=vectors)
            tmp_clustering = AgglomerativeClustering(linkage=AG_linkage, n_clusters=n_clusters, metric=AG_affinity)
            tmp_clustering.fit(vectors)
            tmp_labels = tmp_clustering.labels_
        else:
            tmp_labels = [0]
        return tmp_labels

    def add_data(self, groupname, additional_dataframe, add_embeddings):
        additional_vectors = np.array([])
        if (groupname[0], groupname[1][0]) in self.all_additional_lemmas:
            vec_indx = additional_dataframe[
                (additional_dataframe[self.target_col[0]] == groupname[0]) & (
                    additional_dataframe[self.target_col[1]].str.startswith(groupname[1][0]))].index
            additional_vectors = np.array(add_embeddings)[vec_indx]
        return additional_vectors

    def add_constraint_data(self, groupname, constraint_dataframe, constraint_embeddings):
        if (groupname[0], groupname[1][0]) in self.all_constraint_lemmas:
            additional_dataframe = constraint_dataframe[(constraint_dataframe[self.target_col[0]] == groupname[0])
                                                        & (constraint_dataframe[self.target_col[1]].str.startswith(
                groupname[1][0]))]
            additional_index = additional_dataframe.index.tolist()
            additional_vectors = np.array(constraint_embeddings)[additional_index]
        else:
            additional_dataframe = np.array([])
            additional_vectors = np.array([])
        return additional_dataframe, additional_vectors

    def compute_silhouette_clusters(self, groupdf_size, vectors):
        best_score, best_n = -1, 1
        for n_cl in range(2, SILH_max_cl):
            if n_cl + 1 > groupdf_size:
                break
            tmp_clustering = AgglomerativeClustering(linkage=AG_linkage, n_clusters=n_cl, metric=AG_affinity)
            tmp_clustering.fit(vectors)
            tmp_labels = tmp_clustering.labels_
            scoring = silhouette_score(vectors, tmp_labels)
            if scoring > best_score:
                best_score = scoring
                best_n = n_cl
        n_clusters = best_n
        return n_clusters
