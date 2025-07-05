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

This repository contains the modeling and analysis code used in the following paper:[
**Pan, X., Mohammadi, N., & Taylor, J. (2025). _Exposing biased risk estimation of 
emergency service disruptions during urban floods._**](https://www.researchsquare.com/article/rs-6422955/v2)

## Code structure
The core code is in `main_modeling.py` (which calls `model.py`) and `main_analysis.ipynb` 
(which calls `analysis.py`).

- `main_modeling.py` performs the first stage of computation, 
which estimates ambulance travel time (the indicator of EMS delay risk) 
based on flood modeling results. 
It controls ArcGIS modules using arcpy to implement the routing-based risk estimation.

- `main_analysis.ipynb` handles the second stage of computation, 
presenting the analysis process and results based on the estimated risks. 
The outputs directly correspond to the figures and findings in the paper and the supplementary materials. 

- If `main_analysis.ipynb` is too large to display on GitHub, it is recommended to 
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
- numpy
- pandas
- geopandas
- arcpy (requires license to ArcGIS Pro)
- networkx
- pysal (for regression)
- plotly (for visualization)
- osmnx (for pulling road data only)
- rasterio (for processing flood maps only)
- geojson (for a couple of visualization only)
- simpledbf (for function merge_road_info_VDOT only)

## Data
The data folder 
used in the study is deposited on the author's
[Google Drive](https://drive.google.com/drive/folders/1mxyiUylxluWH87xTQuYMZmrEDUn3v0rs?usp=sharing) 
currently and open for public use. Please organize the data
folder as indicated by the *config_vb.py*.
Please contact with the author if you have any 
question
([xyp@gatech.edu](mailto:xyp@gatech.edu)).


