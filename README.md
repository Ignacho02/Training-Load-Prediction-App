# Training-Load-Prediction-App

 ![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)
 ![Streamlit](https://img.shields.io/badge/streamlit-1.28-orange?logo=streamlit)

 ## Table of Contents

 1. [Overview](#overview)
 2. [Repository Structure](#repository-structure)
 3. [Installation](#installation)
 4. [Model Download](#model-download)
 5. [Usage](#usage)
 6. [Features](#features)
 7. [Dependencies](#dependencies)
 8. [Contact](#contact)

 ---

 ## Overview

 The **Training-Load-Prediction-App** minimizes the gap between prescribed external training load and actual load for football players. By addressing this, the app helps:

 - Optimize player performance  
 - Reduce injury risk  
 - Improve training monitoring and load adjustment

 The repository contains both the **modeling pipeline** and an interactive **Streamlit application**.

 ---

 ## Repository Structure

 | Folder / File | Description |
 |---------------|-------------|
 | `.gitattributes` | Git configuration file for attributes |
 | `Data_generation/` | Scripts to generate synthetic or processed datasets |
 | `Final_model_deployment/` | Deployment scripts and artifacts (chosen model)|
 | `Logo.jpg` | Personal logo |
 | `Modelling/` | EDA, feature engineering, and model development scripts |
 | `README.md` | This file |
 | `app.py` | Streamlit app for interactive predictions |
 | `feature_columns.pkl` | Pickle file storing model feature columns |
 | `output_columns.pkl` | Pickle file storing model output columns |
 | `preprocessor.pkl` | Pickle file for preprocessing pipeline |
 | `synthetic_full_dataset.csv` | Sample dataset for testing |

 > ⚠️ **Note:** The trained model (`final_rf_tuned_fast_model.pkl`) is too large for GitHub and is automatically downloaded from Google Drive at runtime.

 ---

 ## Installation

 1. Clone the repository:

 ```bash
 git clone https://github.com/Ignacho02/Training-Load-Prediction-App.git
 cd Training-Load-Prediction-App
 ```

 2. Create and activate a virtual environment:

 ```bash
 python -m venv venv
 # Windows
 venv\Scripts\activate
 # Mac/Linux
 source venv/bin/activate
 ```

 3. Install dependencies:

 ```bash
 pip install -r requirements.txt
 ```

 ---

 ## Model Download

 The model file exceeds GitHub's file limit. The app automatically downloads it from Google Drive using **gdown**.

 - URL: [Google Drive link](https://drive.google.com/file/d/1Fmw782ET3fxqZphucD-PKrLFjFa6Xccq/view?usp=drive_link)  
 - gdown is included in `requirements.txt`.

 Alternatively, manually download the model and place it in the root folder:

 ```bash
 final_rf_tuned_fast_model.pkl
 ```

 ---

 ## Usage

 1. Launch the Streamlit app:

 ```bash
 streamlit run app.py
 ```

 2. Upload session data or use `synthetic_full_dataset.csv`.  
 3. The app preprocesses data and predicts actual training load.  
 4. View results and optionally export them.

 ---

 ## Features

 - Interactive Streamlit web app  
 - Automated download of large model from Google Drive  
 - Complete preprocessing pipeline (`preprocessor.pkl`)  
 - Prediction outputs aligned with football conditional metrics  
 - Sample dataset for testing

 ---

 ## Dependencies

 - Python 3.10+  
 - pandas  
 - numpy  
 - scikit-learn  
 - joblib  
 - streamlit  
 - gdown

 *(Full list in `requirements.txt`)*

 ---

 ## Contact

 - Author: Ignacio García Bilbao 
 - GitHub: [Ignacho02](https://github.com/Ignacho02)  
 - Email: [nachogarbil@gmail.com](mailto:nachogarbil@gmail.com)
