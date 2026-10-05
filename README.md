# ML Dashboard for Car Price Forecasting

Car price prediction: feature engineering, a comparison of linear, forest, gradient-boosting and MLP models tuned with random search, and a final stacking ensemble (validation MAE about £1,186) deployed as an interactive Streamlit pricing app.

Course project for Machine Learning, MSc in Data Science and Advanced Analytics, NOVA IMS (2025).

## What the project does

1. **Exploration and cleaning** (`1_EDA_Preprocessing.ipynb`): fixes inconsistencies, analyses the variables one by one and in pairs, handles missing values and duplicates, creates new features and encodes categorical ones.
2. **Preprocessing and feature selection** (`2_Holdout_Preproc_featSelect.ipynb`): holdout split, outlier treatment, log-transform of the price, robust scaling and KNN imputation. Filter, wrapper and embedded methods reduce the data to 20 final features.
3. **Modelling** (`3_Modeling.ipynb`): trains and compares eight models, tuned with a custom random search. Models learn the log price; MAE is measured on the original price scale.
4. **Pricing app** (`4_Open_ended_section.ipynb`, `app.py`): packs preprocessing and the final model into one scikit-learn pipeline with custom transformers, served by a Streamlit form that returns a price from a car's brand, model, year, mileage, tax, engine size and fuel data.

## Results

| Model | Validation MAE (£) |
| --- | --- |
| Linear Regression | 2,211 |
| Lasso | 2,244 |
| MLP (four hidden layers) | 1,494 |
| K-Nearest Neighbours | 1,458 |
| Random Forest | 1,240 |
| Gradient Boosting | 1,218 |
| HistGradientBoosting | 1,196 |
| **Stacking ensemble (final)** | **1,186** |

The final model stacks two HistGradientBoosting models and one Gradient Boosting model under a linear-regression meta-model. It produced the price predictions for 32,567 test cars.

## Repository contents

| File | Description |
| --- | --- |
| `delivery_2/1_EDA_Preprocessing.ipynb` | Exploration, cleaning and feature creation |
| `delivery_2/2_Holdout_Preproc_featSelect.ipynb` | Split, preprocessing and feature selection |
| `delivery_2/3_Modeling.ipynb` | Model training, tuning and comparison |
| `delivery_2/4_Open_ended_section.ipynb` | Final pipeline and app build |
| `delivery_2/app.py` | Streamlit pricing app |
| `test_predictions/` | Predictions on the test set |
| `environment.yml` | Conda environment |
| `LICENSE` | MIT licence |

## Tech stack

Python · pandas · scikit-learn (Random Forest, Gradient Boosting, HistGradientBoosting, MLP, Stacking) · Streamlit · conda

## Authors

· Artem Polikarpov · [LinkedIn](https://www.linkedin.com/in/artem-polikarpov-068a4313) · [All projects](https://github.com/apnovaims)

· Diogo Montenegro

· Francisco Martins

· João Cardoso
