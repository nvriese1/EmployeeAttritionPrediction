# People Analytics: Employee Attrition Prediction Tool
===============================

By using anonymized IBM Human Resources (HR) data, I created a tool to help company HR 
departments allocate resources to employees at the greatest risk of attrition (quitting/leaving),
reducing wasted retention efforts on employees with little potential for attrition. 
Through utilization of exploratory data analyis, feature engineering, unsupervised machine learning, 
and hyperparameter optimization, I developed an efficient model to solve this problem.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data    
    │   ├── interim        <- Intermediate data that has been transformed.
    │       └── HR_data_cleaned.csv 
    │   ├── processed      <- The final, canonical data sets for modeling.
    │       └── HR_data_cleaned_EDA.csv
    │       └── features.csv
    │       └── idx_test.csv
    │       └── idx_train.csv
    │       └── X_test.csv
    │       └── X_train.csv
    │       └── y_test.csv
    │       └── y_train.csv
    │   └── raw            <- The original, immutable data dump.
    │       └── employee_survey_data.csv
    │	    └── manager_survey_data.csv
    │	    └── general_data.csv
    │	    └── in_time.csv
    │	    └── out_time.csv 
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   ├── 1.0-hr-data-wrangling.ipynb
    │   ├── 2.0-hr-data-exploration.ipynb
    │   ├── 3.1-hr-indepth-analysis.ipynb
    │   └── 3.2-hr-indepth-analysis.ipynb 
    │
    ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
    │   └── data_dictionary.xlsx
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── FinalProjectReport.pdf
    │   └── FinalProjectPresentation.pptx                 
    │
    └── src                <- Source code for use in this project.
        ├── 1.0-hr-data-wrangling.py
        ├── 2.0-hr-data-exploration.py
        ├── 3.1-hr-indepth-analysis.py
        └── 3.2-hr-indepth-analysis.py

        
