## Exploratory tools
A variety of exploratory analysis tools are available to assess the structure of the dataset.

### Latent space clusters
The densities of each component are obtained from the Gaussian mixture model and used to :
- assign "hard" label to a datapoint based on the dominant component. This label is used to color code the datapoint.
- assign "soft" labels to a datapoint based on each components relative density. The maximum relative density determines the opacity of each data point's color.

### Probability filter
This Gaussian mixture model can be further used to filter the distribution via the {obj}`~lse.libs.dash_core.components_models.PlotControlComponent.PROBA_FILTER` slider. Decreasing slider's high bound filters out data point of decreasing log-probability (i.e., outliers) ; conversely, increasing slider's low bound filters out the most normal data points, hence highlighting outliers.

### Plot clusters
The plotted data can be further analyzed using K-means clustering.

⚠️ Note that that this clustering works on the current view of the data: it only considers the dimensions on plot and current data points, disregarding any underlying structure of the dataset.

### Gini impurity index
When slicing the data along a dimension, this inset displays the [*weighted average gini impurity index*](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity) of the Gaussian model on the partition defined by {obj}`~lse.libs.dash_core.components_models.DropdownComponent.DATA_SLICER`.

This metrics quantifies how well the partition of the data by the {obj}`~lse.libs.dash_core.components_models.DropdownComponent.DATA_SLICER` dimension is reflected in the Gaussian mixture model's clustering.


### Plot tabs
#### Datum reconstruction
When hovering on a data point in the low-dimensional plot, the **"signature"** (i.e., reconstructed signals using PCs capturing *x*% of the variance in the data) is drawn in the form of a radar plot in the `Data reconstruction` tab.

- Radar plot signature can be modified via the {obj}`~lse.libs.dash_core.components_models.DropdownComponent.SIGNATURE` dropdown menu. Variable options are ordered in descending order of their contribution (i.e. *loading*) onto the first selected PC.

- The {obj}`~lse.libs.dash_core.components_models.PlotControlComponent.VARIANCE_FILTER` can be used to control the % of variance retained when fitting the Latent Gaussian model and reconstructing signal.

#### PC loadings
This inset shows a barplot of *loadings* of the original {obj}`~lse.libs.dash_core.components_models.DropdownComponent.SIGNATURE` variables onto the selected PC, sorted by decreasing order of magnitude.

The {obj}`~lse.libs.dash_core.components_models.PlotControlComponent.RESET` button reinitializes the list of {obj}`~lse.libs.dash_core.components_models.DropdownComponent.SIGNATURE` variables with the 3 most contributing variables on the first selected PC.