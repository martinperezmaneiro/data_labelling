# data_labelling
This repository is a module to perform binary and semantic segmentation labelling on the NEXT data. 

The purpose for this labelling is to train a Convolutional Neural Network, more information here: https://github.com/martinperezmaneiro/NEXT_SPARSECONVNET

For a tutorial on how to set up and work with this repository, check the Notion page: https://mpema.notion.site/data_labelling-36c3294fcc5b4ab085d85990b78d2fbb

## Requirements
Needed miniconda and IC (maybe soon not needed, as it uses just a few functions): https://github.com/next-exp/IC

## Organization
The repository can be set running
```
source setup.sh
```
and later import any of the folders and its contents as (for example)
```
import utils.labelling_utils
```
