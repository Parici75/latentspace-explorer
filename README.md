# Latent Space Explorer


<div align="center" markdown="1">

[![Documentation](https://img.shields.io/website?label=docs&url=https://parici75.github.io/latentspace-explorer)](https://parici75.github.io/latentspace-explorer)
![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![Black](https://img.shields.io/badge/Code%20style-Black-black)](https://black.readthedocs.io/en/stable/)
[![linting - Ruff](https://img.shields.io/badge/Linting-Ruff-yellow)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy.readthedocs.io/en/stable/index.html#)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![CI](https://github.com/Parici75/latentspace-explorer/actions/workflows/test.yml/badge.svg)](https://github.com/Parici75/latentspace-explorer/actions/workflows/test.yml)
[![GitHub License](https://img.shields.io/github/license/Parici75/latentspace-explorer)](https://github.com/Parici75/latentspace-explorer/blob/main/LICENSE)

</div>

----------------
[`latentspace-explorer`](https://github.com/Parici75/latentspace-explorer) is a toolbox based on [Dash](https://plotly.com/dash/) that let you easily dive into multivariate data.


![latentspaceexplorer-demo](docs/assets/latentspaceexplorer-demo.gif)


The library was designed as a user-friendly tool for carrying out interactive exploratory data analysis of multidimensional dataset.

Data exploration in Jupyter notebooks can be tedious and lead to a pile of endless variations of statistical visualization plots.

`latentspace-explorer` combines dimensionality reduction, probability density estimation, and clustering techniques to enable deep exploration of multivariate datasets across different facets.

Documentation
-
Features and workflow example can be found in the [documentation](https://parici75.github.io/latentspace-explorer).


Getting started with `latentspace-explorer` on your data
-
Dash being [stateless](https://dash.plotly.com/sharing-data-between-callbacks), `latentspace-explorer` uses [Redis](https://redis.io/solutions/caching/) as a caching backend.

To start a Redis database locally in a Docker container, run:

```bash
make start-redis
```
This pulls the latest Redis Docker image if need be and starts a `Redis` server listening on the default port `6379`.


There are two entry points to the application:

### From a Python script / Jupyter notebook
`latentspace-explorer` can easily be integrated in your python data workflow in a Jupyter environment, via the `LatentSpaceInterface` class.

First, clone and install the package with pip:

```bash
pip install .
```

Starting the application to explore a dataset is as easy as:
```python
import plotly.express as px
from lse import LatentSpaceInterface

df = px.data.iris()
app = LatentSpaceInterface(df, dashboard_title='Iris dataset', port=8050).start()
```

### As a standalone application
`latentspace-explorer` can also be executed as a standalone application.

To build and run the application locally, clone the repository and run:
```bash
make run-app
```

This target builds a `latentspaceexplorer-app` docker image, pulls a `Redis` image and start the containers on `localhost` via docker-compose.

The app will be accessible at `http://localhost:8050/`

Spreadsheets of data can be loaded via the `Ìmport` section.


### Pushing to production
Pushing the app to production requires the following three steps:
1. Setting up a production Redis cache database.
2. Building and pushing a production Docker image to a docker/artifact registry. Refer to the `push-prod-container` target of the `makefile` to accomplish that.
3. Pulling the image and starting a container on a virtual machine or a serverless solution.


## Setting up a development environment using Poetry

First make sure you have Poetry installed on your system (see [instruction](https://python-poetry.org/docs/#installing-with-the-official-installer)).

Then, assuming you have a Unix shell with [`make`](https://edoras.sdsu.edu/doc/make.html), create and set up a new Poetry environment :

```bash
make init
```

To make the Poetry-managed kernel available for a globally installed Jupyter:

```bash
poetry run python -m ipykernel install --user --name=<KERNEL_NAME>
jupyter notebook
```

Requirements
-
- [openTSNE](https://github.com/pavlin-policar/openTSNE/), extensible, parallel implementations of t-SNE.
- [StatsPlot>=0.2.3](https://github.com/Parici75/statsplotly.git), a custom statistical visualization library based on Plotly.
- [MLToolbox>=0.1.2](https://github.com/Parici75/mltoolbox.git), a custom statistical learning toolbox.


Extending the project
-
Adapting this toolbox to a specific use case typically involves:
- setting up a proper data ingestion module to interface with the data source.
- writing the data preprocessing module that prepare the data to be analysed.

[Here](https://benjaminroland.onrender.com/work/interactive-anomaly-detector-and-scorer) is an example of how one can use the core functionalities of this toolbox to tackle real-world customer issues.


Limitations
-
### Working with big data
`latentspace-explorer` does not work with big data, that is, the full dataset needs to fit on the RAM of the machine executing the application.

### Number of points rendered
By default, the number of points displayed is limited to avoid cluttering the UI. Beyond this limit, the most "abnormal" data points (i.e. with the highest negative log-probabilities) are filtered out automatically.


Author
-
[Benjamin Roland](https://benjaminroland.onrender.com/)
