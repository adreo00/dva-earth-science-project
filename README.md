# dva-earth-science-project

## Setup

- create new conda environment using `conda create --name dva-earth-science jupyter python=3.9`
- active conda environment: `conda activate dva-earth-science`
- run `pip install -r requirements.txt`

## Fetch Data

TODO: Instructions here on how to download data if not included in github repo 
- Download data here: https://gtvault-my.sharepoint.com/:f:/g/personal/adreo3_gatech_edu/EqgTY5X4p2dAq1ZPubGyzn4BoEblaknok6zGJ00mgi01VQ?e=0PI207
- save the following files to `data/` directory of repository
  - `MicSigV1.json`
  - (optional) `clean_data.parquet`
    - this allows you to skip the pre-proecessing

## Run notebook 
run: `jupyter notebook` or run the notebook using and IDE like (VSCode)[https://code.visualstudio.com/docs/datascience/jupyter-notebooks]