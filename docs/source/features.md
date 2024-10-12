## Exploratory tools
A variety of exploratory analysis tools are available to assess the structure of the dataset.

### Latent space clusters
The density of each component of the mixture models are obtained from the Gaussian mixture model and used to :
- assign "hard" label to a datapoint based on the dominant component. This label is used to color code the datapoint.
- assign "soft" labels to a datapoint based on each components relative density. The maximum relative density determines the opacity of each data point's color.

### Probability filter
This Gaussian mixture model can be further used to filter the distribution via the `Probability filter` slider. Decreasing slider's high bound filters out data point of decreasing log-probability (i.e., outliers) ; conversely, increasing slider's low bound filters out the most normal data points, hence highlighting outliers.

### Plot clusters
Plotted data can be further explored through K-means clustering.

⚠️ Note that this clustering is designed as a visual inspection tool : it only uses the current dimensions and data points being plotted, independently of the latent distribution of the dataset.

### Gini impurity index
When slicing the data along a dimension, this inset displays the [*weighted average gini impurity index*](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) of the Gaussian model on the partition defined by the dimensions `data slicer`.

This metrics quantify how well the partition of the data by the `data slicer` dimension is reflected in the Gaussian mixture model's clustering.


### Plot tabs
#### Datum reconstruction
When hovering on a data point in the low-dimensional plot, the **"signature"** (i.e., reconstructed signals using PCs capturing x% of the variance in the data) is drawn in the form of a radar plot in the `Data reconstruction` tab.

- Radar plot signature can be modified via the `Signature variables` dropdown menu. Variable options are ordered in descending order of their contribution (i.e. *loading*) onto the first selected PC.

- The `Variance to retain` slider can be used to control the % of variance retained when fitting the Latent Gaussian model and reconstructing signal.

#### PC loadings
This inset shows a barplot of *loadings* of the original `Signature variables` onto the selected PC, sorted by decreasing order of magnitude.

The `reset` button reinitializes the list of `Signature variables` with the 3 most contributing variables on the first selected PC.