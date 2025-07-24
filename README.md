# Exposing biased risk estimation of emergency service disruptions during urban floods

## Introduction
Emergency service (ES) disruption is a major impact of urban flooding. 
As a public safety and health safeguard that requires equity, ES disruption risk mitigation 
must examine whether socially vulnerable neighborhoods 
(e.g., low-income groups and minorities) are facing disproportionately higher risks.
Assuming estimated risk is a good indicator of the real risk levels, 
extant research and practice overlook the potential biases in risk estimation 
regarding social vulnerability. 
Greater risk underestimation for more vulnerable populations could lead 
to biased risk estimation and less effective risk mitigation.
We implemented the routing-based risk estimation, 
calculated errors against a real-world ES dataset, and analyzed the relationship 
between risk underestimation and social vulnerability using spatial regression. 
It is found that the routing-based approach underestimates the ES delay risks faced 
by low-income communities while high-income communities are better off 
in the case of 2016 Hurricane Matthew in Virginia Beach. 
This bias can be partially explained by the potentially lower ES capacity and
degraded traffic conditions in low-income communities.

This repository contains the modeling and analysis code used in the following pre-print:[
**Pan, X., Mohammadi, N., & Taylor, J. (2025). _Exposing biased risk estimation of 
emergency service disruptions during urban floods._**](https://www.researchsquare.com/article/rs-6422955/v2)

## Code structure
The core code is in `main_modeling.py` (which calls `model.py`) and `main_analysis.ipynb` 
(which calls `analysis.py`).

- `main_modeling.py` performs the first stage of computation, 
which estimates ambulance travel time (the indicator of EMS delay risk) 
based on flood modeling results. 
It controls ArcGIS modules using arcpy to implement the routing-based risk estimation. This script could be skipped
without influencing running the following scripts.

- `main_analysis.ipynb` handles the second stage of computation, 
presenting the analysis process and results based on the estimated risks. 
The outputs directly correspond to the figures and findings in the paper and the supplementary materials. 
Running this script requires access to ArcGIS Pro 3. 
Computation in most of the cells should be finished within at most several minutes.
The major expected results have been included in the notebook.

- `main_analysis.ipynb` is too large to display on GitHub. It is recommended to 
enter the notebook [address](https://github.com/pppxiyu/EquiRisk/blob/main/main_analysis.ipynb) into [nbviewer](https://nbviewer.org/) to render the notebook.

The `utils` folder contains code for data preprocessing, regression analysis, and visualization.

- Data preprocessing modules:
  - `preprocess_incidents.py`: processes EMS incident data
  - `preprocess_roads.py`: processes urban road network data
  - `preprocess_station.py`: processes EMS station location data
  - `preprocess_graph.py`: processes the generated road network graph data

- Regression and visualization modules:
  - `regression.py`: conducts regression analysis
  - `visualization.py`: generates visualizations

To use the `visualization` module, 
please place your Mapbox token in a plain text file named `mapboxToken.txt`.

## Requirements
- Python (3.11.10)
- numpy (1.24.3)
- pandas (2.0.3)
- geopandas (1.1.1)
- arcpy (3.4) and ArcGIS Pro (3.4.2, requiring license)
- networkx (3.3)
- pysal (24.1, for regression)
- plotly (5.24.1, for visualization)
- osmnx (2.0.3, for pulling road data only)
- rasterio (1.3.10, for processing flood maps only)
- geojson (3.2.0, for a couple of visualization only)
- simpledbf (0.2.5, for function merge_road_info_VDOT only)
- Please manually Configure dependencies and ArcGIS Pro. It may take 30 to 60 min.

## Data
The [data folder](https://drive.google.com/drive/folders/1mxyiUylxluWH87xTQuYMZmrEDUn3v0rs?usp=sharing) and the [GIS model](https://drive.google.com/drive/folders/1gkkItIlMaQidPVKT2WL_V9FNeKBpZ5D6?usp=sharing) 
 involved in the computation are deposited on the author's Google Drive.
Please carefully organize the data
folder as indicated by the `config_vb.py`. 
To run this program with custom data, please replace all files in `config.py`
and ensure or convert the data format to match the format currently used by the program.
Please contact with the author if you have any 
question
([xyp@gatech.edu](mailto:xyp@gatech.edu)).

## Run
We recommend running `main_analysis.ipynb` cell by cell. 
This file contains the reproducible main analysis process and results used in the study. 
If readers are interested in modeling details such as routing, 
please uncomment the relevant code in `main_modeling.py` to run it. 
Please note that `main_modeling.py` is not designed to be executed from start to finish as a single script.

