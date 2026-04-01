# ML: Project1
-------------------------------------------
### End-to-End ML Project:

This project aims to cover the formal project structure of a prediction of Student performance based on data it is trained on (Student_Performance.csv- The data consists of 8 column and 1000 rows): Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetld=74977


### Problem Statement:
End to End Flask based prediction of score (math scores affected by other variables such as Gender, Ethnicity,
Parental level of education, Lunch and Test preparation course) following industry standard code structure and practices.

### Approach/ Process Followed:
1. Project structure definition and repo establishment
2. Setting up Logging and CustomExceptions
3. Import dataset
4. EDA on imported Dataset
5. Basic Model training on dataset and selection of Target variable
7. building utils and components:
    + data_ingestion
    + data_transformation
    + model_training
8. establishing pipeline
9. basic UI + Flask setup
10. Testing and deploying

### Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open in browser:
- http://127.0.0.1:5000/
- http://127.0.0.1:5000/predict

### Requirements:
```text
    catboost          1.2.1
    dill              0.4.1
    Flask             3.1.3
    ipykernel         7.2.0
    matplotlib        3.10.8
    matplotlib-inline 0.2.1
    numpy             2.2.6
    pandas            2.3.3
    pillow            12.1.1
    pip               26.0.1
    scikit-learn      1.7.2
    scipy             1.15.3
    seaborn           0.13.2
    xgboost           3.2.0
```

### Results/ Scores (Output Score and Selected Model):

From the latest successful training cycle (log timestamp 2026-04-01 16:59:52):

- Selected model: Lasso
- Best R2 score: 0.8806548187533167
- MAE: 4.20951972955114
- MSE: 29.041265452277912
- RMSE: 5.388994846191441
- Best parameters: selection=random, max_iter=1000, alpha=0.01

### Note:
- Albeit a basic Regression based ML project, said project is treated and built with industry standard techniques and practices to ensure optimal file structure and workflow with detailed logs.
- Trained model and preprocessing objects are serialized using `dill` in the `artifacts/` directory.
- Generated training/evaluation reports are stored in `artifacts/reports/`.
- Prediction pipeline loads the serialized preprocessor/model to run inference for the Flask app.


### Project-Structure:

```text
MLProject1/
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
├── setup.py
├── artifacts/
│   ├── data.csv
│   ├── test.csv
│   ├── train.csv
│   └── reports/
│       ├── *.csv
│       ├── *.json
│       └── regression_plots/
├── notebook/
│   ├── EDA_STUDENT_PERFORMANCE.ipynb
│   ├── MODEL_TRAINING.ipynb
│   └── data/
│       ├── Student_Performance11.csv
│       └── StudentsPerformance.csv
├── src/
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   └── pipeline/
│       ├── __init__.py
│       ├── predict_pipeline.py
│       └── train_pipeline.py
├── templates/
│   ├── home.html
│   └── index.html
└── venv/
```


### Data Dictionary

| Feature | Type | Example | Description |
|---|---|---|---|
| gender | categorical | female | Student gender |
| race_ethnicity | categorical | group C | Student ethnicity group |
| parental_level_of_education | categorical | bachelor's degree | Parent education level |
| lunch | categorical | standard | Lunch type |
| test_preparation_course | categorical | completed | Test prep course status |
| reading_score | numeric | 72 | Reading score |
| writing_score | numeric | 74 | Writing score |

Target predicted by app:
- `math_score` (regression output)

### Way To Setup an API

Current Flask routes:
- `GET /` -> loads the landing page (`index.html`)
- `GET /predict` -> loads prediction form (`home.html`)
- `POST /predict` -> accepts form fields and returns predicted math score

Run API server:

```bash
python app.py
```

Submit prediction (form-style request):

```bash
curl -X POST http://127.0.0.1:5000/predict \
    -d "gender=female" \
    -d "ethnicity=group C" \
    -d "parental_level_of_education=bachelor's degree" \
    -d "lunch=standard" \
    -d "test_preparation_course=completed" \
    -d "reading_score=72" \
    -d "writing_score=74"
```

### High-level pipeline flow:

1. Data ingestion reads source dataset and performs train/test split (random_state taken as: 42).
2. Data transformation builds preprocessing pipelines for following cols:
   - Numerical: reading score, writing score
   - Categorical: gender, race/ethnicity, parental level of education, lunch, test preparation course
3. Baseline training evaluates multiple regressors and saves baseline score report + comparison plot. (For eval)
4. Hyperparameter tuning runs for tuned candidates (Ridge, Gradient Boosting, Lasso, CatBoost; or the top 5 best model based on r2 score at the time of training), then logs MAE/MSE/RMSE/R2 (as per eval function (refer utils.py)) for each.
5. Tuned scores are then compared; best model is selected by highest R2 and persisted for inference and further usage.
6. Predict pipeline loads persisted preprocessor + model and serves predictions via Flask.

### Latest tuning outcome summary:

- Lasso: R2 0.8806548187533167 (selected)
- Ridge: R2 0.8804480338276083
- CatBoost: R2 0.8728678562724155
- Gradient Boosting: R2 0.8680342160816621

#### PS:
- this a project made majorly to understand the ins and outs of an end to end ML project, future plans for this include a decent deployment + a basic Docker image for containerization

