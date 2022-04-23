# dva-earth-science-project

## Setup

- create new conda environment using `conda create --name dva-earth-science jupyter python=3.9`
- active conda environment: `conda activate dva-earth-science`
- run `pip install -r requirements.txt`

## Fetch Data

- Download data here: https://gtvault-my.sharepoint.com/:f:/g/personal/adreo3_gatech_edu/EqgTY5X4p2dAq1ZPubGyzn4B6-Q5IhVFUCOA8gAH-je00g?e=aSVPCq
- save the following files to `data/` directory of repository
  - `MicSigV1.json`
  - (optional) `clean_data.parquet`
    - this file allows you to skip the pre-proecessing notebook (`1etl.ipynb`), and proceed directly to `2model.ipynb`

## Run notebooks
run: `jupyter notebook` or run the notebook using and IDE like (VSCode)[https://code.visualstudio.com/docs/datascience/jupyter-notebooks]

## Explanation of notebooks

- `1etl.ipynb`: begins with `MicSigV1.json`. Performs basic tranformation and decomposition on raw seismic data
- `2model.ipynb`: begins with `clean_data.parquet`, builds several different earthquake classification models, and compares performance of each.

## Run Visualizations
- to begin serving the visual, run the following command
  -  `python -m http.server`
- next, access the visual in your web browser. By default, this is accessible at:`http://localhost:8000/Visualization/earthquakes.html`