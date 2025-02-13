# LinearSVC Model Project

This project implements a Linear Support Vector Classification (LinearSVC) model for impact classification.

## Prerequisites

- Python 3.8

### train and test data from Auswirkung/data_prep project

- make sure the data_prep project is in same folder like this (linear_svc) project 
- this project refers to outputed data from data_prep poject located in `data_prep/src/_2data_final`

## Setting up the Environment

1. **Create a virtual environment:**

  ```
  python -m venv env
  ```

2. **Activate the virtual environment:**

- On Windows:
  ```
  env\Scripts\activate
  ```

- On Unix or macOS:
  ```
  source env/bin/activate
  ```

## Installation

1. **Clone the repository:**
git clone git@gitlabdca.itsarz.bwi:coki/impact_analyse/klassifikationauswirkung/linear_svc.git
2. **Navigate to the project directory:**
cd linear_svc
3. **Install the required packages:**  
``pip install -r requirements.txt``  


## Project Structure

```
project_root/
│
├── src/
│   ├── train.ipynb
│   ├── data_output/
│   │   └── val_data.csv
│   ├── interpretation/
│   │   └── [interpretation notebooks]
│   └── val_test/
│       ├── val.ipynb
│       ├── test.ipynb
│       └── model_surence.ipynb
│
├── output/
│   ├── feature_names.joblib
│   ├── model.joblib
│   └── vectorizer.joblib
│
└── README.md
```

## Training

The model is trained using the `train.ipynb` notebook in the `src` folder. This notebook uses `train.csv` from the `data_prep` project as input data.

### Outputs
After training, the following files are saved in the `output` folder:
- `feature_names.joblib`: Names of the features used in the model
- `model.joblib`: The trained LinearSVC model
- `vectorizer.joblib`: The fitted vectorizer

## Validation

Validation is performed in the `val.ipynb` under the `src/val_test/` directory. It uses the `val_data.csv` generated during the training process and stored in `src/data_output/`.

## Testing

Testing is conducted using the `test.ipynb` in the `src/val_test/` directory. It uses test data from the `data_prep` project.

## Model Sureness Analysis

Model sureness is analyzed in the `model_surence.ipynb` notebook located in the `src/val_test/` directory.

## Interpretation 

The interpretation of the model is done using notebooks in the `src/interpretation/` folder. It uses `feature_names.joblib` to combine coefficients from the model for a readable interpretation of the model output.

### Interpretation Details
- Features contributing up to 50% of the model's explainability are included in the interpretation.