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

### Note:
+ Albeit a basic Regression based ML project, said project is treated and build with industry standard techniques and practices to ensure optimal file structure and workflow with detailed logs.

### Project-Structure:

```text
MLProject1/
├── app.py
├── README.md
├── requirements.txt
├── setup.py
├── artifacts/
│   ├── data.csv
│   ├── test.csv
│   ├── train.csv
│   └── reports/
│       ├── baseline_model_scores_20260330_002615.csv
│       ├── baseline_model_scores_20260330_003903.csv
│       ├── baseline_model_scores_20260330_012208.csv
│       ├── baseline_model_scores_20260330_182327.csv
│       ├── best_model_summary_20260330_004122.json
│       ├── best_model_summary_20260330_012324.json
│       ├── best_model_summary_20260330_182439.json
│       ├── tuned_model_scores_20260330_002700.csv
│       ├── tuned_model_scores_20260330_004121.csv
│       ├── tuned_model_scores_20260330_012324.csv
│       ├── tuned_model_scores_20260330_182439.csv
│       └── regression_plots/
├── catboost_info/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   ├── learn/
│   │   └── events.out.tfevents
│   └── tmp/
├── logs/
├── ML_Project_1.egg-info/
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── notebook/
│   ├── EDA_STUDENT_PERFORMANCE.ipynb
│   ├── MODEL_TRAINING.ipynb
│   ├── catboost_info/
│   │   ├── catboost_training.json
│   │   ├── learn_error.tsv
│   │   ├── time_left.tsv
│   │   ├── learn/
│   │   │   └── events.out.tfevents
│   │   └── tmp/
│   └── data/
│       ├── Student_Performance11.csv
│       └── StudentsPerformance.csv
└── src/
    ├── __init__.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    ├── components/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_training.py
    └── pipeline/
        ├── __init__.py
        ├── predict_pipeline.py
        └── train_pipeline.py
```
