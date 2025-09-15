# 🏠 Boston Housing Price Prediction

A *Python machine learning project* to predict Boston housing prices. Easily *train, test, and run predictions* using prepared scripts and notebooks.

---

## Features

- Train multiple regression models (Linear Regression, etc.)  
- Save and load trained models (.pkl) for quick predictions  
- Predict housing prices using Python scripts or Jupyter notebooks  
- Explore dataset and model performance in notebooks  
- Clean and organized project structure  

---

## Project Files

- `app.py` – Main application to run predictions  
- `app-Train-Test.py` – Script for training and testing models  
- `train_boston_models.py` – Train selected models for Boston dataset  
- `train_boston_models_all.py` – Train all models for Boston dataset  
- `Linear Regression_Train-Save-Test.ipynb` – Notebook for linear regression experiments  
- `Project_Regression_Analysis_With_Boston_Housing_Dataset.ipynb` – Full project analysis notebook  
- `data.csv` – Dataset used for training  
- `boston_linear_regression.pkl` – Saved linear regression model  
- `boston_best_model.pkl` – Saved best-performing model  
- `feature_columns.json` – Feature metadata for models  
- `linear_feature_meta.json` – Linear regression feature metadata  
- `requirements.txt` – Python dependencies  
- `model_card.md` – Project documentation  

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/manya9849/Boston-Housing.git
cd Boston-Housing

# (Optional) Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Run scripts
Train all models:

bash
Copy code
python train_boston_models_all.py
Run prediction app:

bash
Copy code
python app.py
Open and experiment in Jupyter notebook:

bash
Copy code
jupyter notebook "Linear Regression_Train-Save-Test.ipynb"
Deliverables
Trained regression models for Boston Housing dataset

Saved model files (.pkl) for fast predictions

Scripts and notebooks for training, testing, and predicting

Dataset and feature metadata included

Learning Outcomes
Understand regression algorithms and their implementation in Python

Work with datasets and preprocessing for machine learning

Save and load models for real-world applications

Use Jupyter notebooks for analysis and experimentation

Organize project files for a clean, reproducible workflow

