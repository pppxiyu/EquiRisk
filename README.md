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
- arcpy (requires license to ArcGIS Pro)
- networkx
- pysal (for regression part)
- plotly (for visualization only)
- osmnx (for pulling road data only)
- rasterio (for processing flood maps only)
- geojson (for a couple of visualization only)
- simpledbf (for function merge_road_info_VDOT only)
## Run
Run each part of main_vb.py as needed by analysis. 
All data used are public dataset. The data folder 
used in the study is deposited on the author's
[Google Drive](https://drive.google.com/drive/folders/1mxyiUylxluWH87xTQuYMZmrEDUn3v0rs?usp=sharing) 
currently and open for public use. Please organize the data
folder as indicated by the *config_vb.py*.
Please contact with the author if you have any 
question
([xyp@gatech.edu](mailto:xyp@gatech.edu)).

The main analysis results are in the file *main_analysis.ipynb*.
If the notebook file is too large to display properly on GitHub, 
it is recommended to enter the notebook [address](https://github.com/pppxiyu/EquiRisk/blob/main/main_analysis.ipynb)
into [nbviewer](https://nbviewer.org/)
to render the notebook.
