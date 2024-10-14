# Workflow

A typical workflow runs as follow :

1. **Load** the data to be explore
    Data must be supplied :
    - To the {class}`~lse.LatentSpaceInterface` class in the form of a *n_observations x n_variables* {obj}`pandas.DataFrame`.
    - As CSV file(s) via the {obj}`~lse.libs.dash_core.components_models.DataLoadingComponent.UPLOAD_DATA` interface.

    🎉 Calendar features are automatically extracted from {obj}`pandas.DatetimeIndex`.


2. **Select** the variables (a.k.a. features) to be included in the analysis. All numeric columns available in the DataFrame are preselected.

    💡 Data can be standardized prior to Principal Components Analysis. Note that while standardization is mandatory for variance analysis when feature dynamic range are not on the same scale, it can be detrimental on some dataset by inflating low variance components, and therefore noise, in the data.

    ⛳ Set the `Number of clusters` to a value which makes sense given your hypothesis on the data generation process.


3. **Run model pipeline**: A decomposition of the dataset in Principal Components is computed with the selected variables, and a mixture of multivariate normal distributions is fitted using data projections onto principal component axes.

    Gaussian Mixture hyperparameters are tuned automatically via a (n_components, covariance_type) grid parameter search to minimize the *Bayesian information criterium*.

    The probability of each data point under the gaussian mixture model is computed.


4. **Explore** data :
    - Select the low-dimensional plot to display.

    - By default, the **negative log-probability** of datapoints is color-coded. Original identifier variables can be used to slice and style the low-dimensional scatter plot.


5. **Export** the data :
    The `Export` section let you export:
    - The matrix of reconstructed data as a CSV file
    - A pickled {class}`~lse.libs.backend.AppBackend` object exposing the data, processing and plotting methods to be used offline.
