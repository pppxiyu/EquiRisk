# Inequitable emergency service flood risk estimation
## Abstract
Emergency service (ES) disruption risk mitigation 
must consider the equity of risk across 
populations with varying level of social vulnerability. 
A key prerequisite is estimating the risk 
distributions. However, prior studies
overlook the potential biases in 
risk estimation errors regarding vulnerability. 
This repo implements the routing-based risk estimation 
and investigating the error against a real-world 
ES dataset, we found evidence of inequitable risk 
underestimation.
## Requirements
- numpy
- pandas
- geopandas
- arcpy (requires access to ArcGIS Pro)
- networkx
- rasterio (for processing flood map only)
- pysal (for regression part)
- plotly (for visualization only)
## Run
Run each part of main_vb.py as needed by analysis. 
All data used are public dataset. 
Please refer to paper for dataset details or 
contact with the author 
([xyp@gatech.edu](mailto:xyp@gatech.edu)).