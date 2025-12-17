# airbnb-honolulu-pricing
Pricing analysis and market segmentation for Airbnb listings in Honolulu, Hawaii.

This repository contains the code and report for the course project on Airbnb price prediction in Honolulu.  
The project explores baseline predictive models and market segmentation strategies to improve pricing accuracy.

## Repository Structure

- `Baseline_Model.py`  
  Implements baseline pricing models, including linear models and tree-based models (Random Forest, LightGBM, CatBoost, XGBoost), used as benchmarks.

- `Honolulu_clustering.py`  
  Performs market segmentation using clustering methods based on geographic and listing characteristics.

- `Cluster_then_predict.py`  
  Implements the cluster-then-predict strategy by fitting separate models within each identified cluster.

- `Cluster_As_Features.py`  
  Implements the cluster-as-features strategy by incorporating cluster membership as additional features in global models.

- `report.pdf`  
  Final project report describing methodology, results, and interpretation.

## Notes

This repository is intended for academic course use.  
All results and figures referenced in the report are generated from the code provided above.

