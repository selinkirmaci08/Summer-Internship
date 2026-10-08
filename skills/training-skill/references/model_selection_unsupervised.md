# Model Selection — Unsupervised Learning

Source: [scikit-learn Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html), linked pages (`clustering.html`, `mixture.html`), and the [algorithm cheat-sheet](https://scikit-learn.org/stable/machine_learning_map.html). Scope: clustering — the unsupervised task most likely to be a genuine *training* deliverable in a classical-ML pipeline. (Manifold learning, biclustering, covariance estimation, novelty/outlier detection, and density estimation exist in scikit-learn but are not deep-dived here; consult the corresponding scikit-learn page directly if a task specifically calls for one of them. Manifold learning specifically has been added to the shared preprocessing reference instead — see below.)

Dimensionality reduction / matrix factorization (PCA, NMF, ICA, etc.) used to live in this file, but it's just as often a feature-engineering step applied *before* another model as it is a standalone deliverable — it has been moved to `../../shared-references/preprocessing_and_feature_engineering.md` §3 so the EDA/preprocessing/feature-engineering skill can own and reuse it. Pull it from there even when dimensionality reduction is the actual training target.

## 1. Clustering

### 1.1 Quick decision table

| Condition | Recommended algorithm | Alternative |
|---|---|---|
| Known cluster count, large dataset, roughly spherical clusters | K-Means (or MiniBatchKMeans for extreme scale) | Bisecting K-Means |
| Unknown cluster count, <10K samples | Mean Shift, VBGMM (Variational Bayesian Gaussian Mixture) | Affinity Propagation |
| Unknown cluster count, larger data | HDBSCAN | OPTICS |
| Non-convex/arbitrary cluster shapes | DBSCAN, HDBSCAN, Spectral Clustering | Hierarchical (single/average linkage) |
| Clusters of varying density in the same dataset | HDBSCAN, OPTICS | — |
| Need to identify/remove outliers as part of clustering | DBSCAN, HDBSCAN | — |
| Image segmentation / graph partitioning | Spectral Clustering | Hierarchical |
| High-dimensional data | MiniBatchKMeans | Spectral (after dimensionality reduction) |
| Very large n_samples | DBSCAN, OPTICS, Bisecting K-Means | MiniBatchKMeans |
| Need connectivity/structural constraints (e.g. only merge spatially adjacent points) | Hierarchical Agglomerative (with `connectivity` matrix) | — |
| Need a hierarchical/dendrogram output | Hierarchical Agglomerative, HDBSCAN, OPTICS, Bisecting K-Means | — |
| Large n_clusters and n_samples, want data reduction as a side effect | BIRCH | Bisecting K-Means |

### 1.2 Algorithm reference

| Algorithm | Cluster shape assumption | Key hyperparameters | Scalability | Notes |
|---|---|---|---|---|
| K-Means | Convex, isotropic (equal variance) | `n_clusters`, `init` (`k-means++`), `max_iter` | Very large n_samples, medium n_clusters | Needs scaled input features (see `../../shared-references/preprocessing_and_feature_engineering.md` §1); sensitive to outliers; minimizes inertia; prone to local minima without `k-means++` |
| MiniBatchKMeans | Same as K-Means | `n_clusters`, `batch_size` | Extreme scale | Slightly worse quality than K-Means, much faster |
| DBSCAN | Density-based, arbitrary shape | `eps`, `min_samples`, `metric` | Very large n_samples | `eps` is critical — too small labels everything noise, too large merges clusters; use a k-distance graph to find the "knee" |
| HDBSCAN | Density-based, variable density | `min_samples`, `min_cluster_size` | Large n_samples | Fixes DBSCAN's single-density limitation; no `eps` needed |
| OPTICS | Density-based, variable density | `max_eps`, `xi`, `metric` | Very large n_samples, large n_clusters | Explores a range of densities instead of one `eps`; produces a reachability plot |
| Hierarchical Agglomerative | Flexible (depends on linkage) | `n_clusters`/`distance_threshold`, `linkage` (`ward`/`complete`/`average`/`single`) | Large n_samples & n_clusters | `ward` behaves like K-Means; `single` can "chain"; supports connectivity constraints |
| Bisecting K-Means | Convex, flat | `n_clusters`, `bisecting_strategy` | Very large n_samples, medium n_clusters | More efficient than K-Means for large n_clusters; more uniform cluster sizes; no empty clusters |
| Spectral Clustering | Non-flat, complex manifolds | `n_clusters`, `affinity`, `assign_labels` | Medium n_samples, small n_clusters | Good for few clusters with non-convex geometry; expensive at scale |
| Mean Shift | Non-flat, smooth density blobs | `bandwidth` | Not scalable with n_samples | Automatically determines number of clusters; can label new samples |
| Affinity Propagation | Flexible size/shape | `preference`, `damping` | Not scalable (O(N²)) | Number of clusters emerges from the data instead of being specified |
| BIRCH | Euclidean, data-reduction oriented | `threshold`, `branching_factor`, `n_clusters` | Large n_clusters & n_samples | Not recommended above ~20 features — prefer MiniBatchKMeans there |
| Gaussian Mixture (GMM) | Gaussian blobs, soft/probabilistic assignment | `n_components`, `covariance_type` | Small-medium n_samples | Gives per-point cluster probabilities instead of hard labels; you must specify `n_components` |
| Variational Bayesian GMM (VBGMM) | Same as GMM | `n_components` (upper bound), `weight_concentration_prior` | Small-medium n_samples (<10K per sklearn's cheat-sheet) | Places a Dirichlet-process-style prior over mixture weights so it can effectively prune unused components — closest scikit-learn equivalent to "infer the cluster count for me" |

### 1.3 Clustering evaluation metrics

| Metric | Needs ground truth? | Range | Use when |
|---|---|---|---|
| (Adjusted) Rand Index | Yes | [-0.5, 1] (adjusted) | Labels available, want chance-corrected agreement |
| (Adjusted/Normalized) Mutual Information | Yes | AMI: [-0.5, 1] | Labels available, want an information-theoretic agreement score |
| Homogeneity / Completeness / V-measure | Yes | [0, 1] | Labels available, want interpretable purity-style breakdown |
| Silhouette Score | No | [-1, 1] | No ground truth; want an intuitive, bounded cluster-quality score (assumes roughly convex clusters, expensive on large N) |
| Davies-Bouldin Index | No | [0, ∞), lower is better | No ground truth; cheaper alternative to silhouette |
| Calinski-Harabasz Index | No | higher is better | No ground truth; ratio of between- to within-cluster variance |
| Inertia (elbow method) | No | [0, ∞), lower is better | K-Means specifically; used to pick `n_clusters` via the "elbow" |

## 2. Dimensionality Reduction / Matrix Factorization

Moved to `../../shared-references/preprocessing_and_feature_engineering.md` §3 (see the note at the top of this file for why).
