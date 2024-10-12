# Latent space models

`latentspace-explorer` orchestrates and combines three different flavours of latent space models:

- [Principal component analysis](https://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/) is a linear dimensionality reduction technique used to remap original datapoints along "principal axes" best describing the variance in the data.

    Principal components' embeddings can be thought of as latent variables composed of linear combinations of the original variables which are selected to maximize the variance of the data in that projection.

    In `latentspace-explorer`, PCA is used both as a *dimensionality reduction* algorithm for visualization purpose, and a *signal tuning* technique to filter out components of low variance in the data.

- [T-distributed Stochastic Neighbor Embedding](https://lvdmaaten.github.io/tsne/) is a non-linear technique aimed at preserving the relative distant between data points. Compared to PCA, it handles outliers better and preserve the local structure in the data.

    The t-SNE's embeddings can be thought of as latent variables encoding a discontinuous proximity map capturing proximal similarity of data points in their original coordinates.

    In `latentspace-explorer`, t-SNE is used as a *dimensionality reduction* algorithm for visualization purpose, complementary to the principal component analysis approach.

- [Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html) are probabilistic models that assume the datapoints are generated from a mixture of Gaussian distributions (or components).

    The gaussian components can be thought of as latent variables, which, in contrast to the example above, take the form of a *probability* for a data point to originate from the corresponding gaussian distribution.

    In `latentspace-explorer`, Gaussian Mixture Models are used for *smooth clustering* and performing *multivariate anomaly quantification*.


## Additional References
[Making sense of Principal Component Analysis](https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues), a concise yet enlightning introduction to Principal Component Analysis on *stackoverflow*.

[t-SNE explained](https://medium.com/swlh/t-sne-explained-math-and-intuition-94599ab164cf), by Achinoam Soroker on *Medium*.

[In Depth: Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html), by Jake VanderPlas.
