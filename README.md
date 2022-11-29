# People Analytics: Employee Attrition Prediction Tool

![SalaryVsWorkedHours](/assets/SalaryVsWorkingHours.png)<br />

## Table of Contents

- [Overview](#overview)
- [Project Organization](#project-organization)
- [Built With](#built-with)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview

 &nbsp;&nbsp;&nbsp;&nbsp;By using anonymized IBM Human Resources (HR) data, I created a tool to help company HR 
departments allocate resources to employees at the greatest risk of attrition (quitting/leaving),
reducing wasted retention efforts on employees with little potential for attrition. <br />
 &nbsp;&nbsp;&nbsp;&nbsp;Through utilization of exploratory data analyis, feature engineering, unsupervised machine learning, 
and hyperparameter optimization, I developed an efficient model to solve this problem.

## Project Organization
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
    │   ├── 3.0-hr-pre-processing.ipynb
    │   └── 4.0-hr-modeling.ipynb
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
        ├── 3.0-hr-pre-processing.py
        └── 4.0-hr-modeling.py

### Built With

<a><button name="button">`Python`</button></a> <br />
<a><button name="button">`Jupyer Notebook`</button></a> <br />
<a><button name="button">`Scikit-Learn`</button></a> <br />
<a><button name="button">`Pandas`</button></a> <br />    

## Contact

Noah Vriese<br />
Email: noah@datawhirled.com<br />
Github: [nvriese1](https://github.com/nvriese1)<br />
LinkedIn: [noah-vriese](https://www.linkedin.com/in/noah-vriese/)<br />
Facebook: [noah.vriese](https://www.facebook.com/noah.vriese)<br />
Twitter: [@nvriese](https://twitter.com/nvriese)<br />

## Acknowledgements

IBM (International Business Machines)<br />
Liscense: MIT
