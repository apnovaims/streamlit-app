# ML Dashboard for Car Price Forecasting

Interactive machine-learning dashboard that predicts car prices through a Streamlit front-end, packaged with a reproducible conda environment.

Course project for Machine Learning, MSc in Data Science and Advanced Analytics, NOVA IMS (2025).

## What the project does

- Trains regression models to forecast car prices.
- Serves predictions through an interactive Streamlit dashboard.
- Ships with a conda environment file so the app can be reproduced on another machine.

## Repository contents

| File | Description |
| --- | --- |
| `delivery_2/` | Project delivery: notebooks and the Streamlit app |
| `test_predictions/` | Predictions on the test set |
| `environment.yml` | Conda environment |
| `LICENSE` | MIT licence |

## How to run

```bash
conda env create -f environment.yml
conda activate <environment name from environment.yml>
streamlit run <path to the app file in delivery_2>
```

## Tech stack

Python · scikit-learn · Streamlit · conda

## Author

Artem Polikarpov · [LinkedIn](https://www.linkedin.com/in/artem-polikarpov-068a4313) · [All projects](https://github.com/apnovaims)
