![Python 3.13.12](https://img.shields.io/badge/python-3.13.12-blue.svg)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)  ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

Data source : https://physionet.org/content/challenge-2012/1.0.0/

# Background provided
## Dataset
- 12000 ICU stay records for adults in csv format
- mix of cardiac, medical, surgical & trauma 
- up to 42 variables recorded - 6 general descriptors, remaining time-series (with multiple observations possible for each)
- general descriptors
    1. RecordID (int)
    2. Age (yrs)
    3. Gender - 0:female; 1:male
    4. Height (cm)
    5. ICUType - 1: Coronary Care Unit, 2: Cardiac Surgery Recovery Unit, 3: Medical ICU, or 4: Surgical ICU
    6. Weight (kg) **NOTE**: weight can be both general descriptor and time series variable 
- time stamps indicate time elapsed since admission and in format hh:mm
- values are expected to be non-negative, with `-1` indicating missing or unknown data 

## Outcome
- Descriptors

    - RecordID (defined as above)
    - SAPS-I score (Le Gall et al., 1984)
    - SOFA score (Ferreira et al., 2001)
    - Length of stay (days)
    - Survival (days)
    - In-hospital death (0: survivor, or 1: died in-hospital)

- Survival definition & constraint
    - Survival > Length of stay  ⇒  Survivor
    - Survival = -1  ⇒  Survivor
    - 2 ≤ Survival ≤ Length of stay  ⇒  In-hospital death

## Challenge

- Predict 0: survival, or 1: in-hospital death
- An estimate of the risk of death (as a number between 0 and 1, where 0 is certain survival and 1 is certain death)

# Data exploration

## Questions
- How many patients in dataset A?
- Any entries missing general descriptor fields i.e. standard intake tests not done?
- Any patients visit ICU more than once?

## Observations
- General descriptors have timestamp 00:00
- Three patients with recordID 140501.0, 140936.0, 141264.0 don't have any other measurements other than general descriptors
- MechVent has zero variance (value 1.0), but is not recorded in all patients. It signifies mechanical ventilation, so could be important, hence should be transformed into presence/absence
- TroponinI, Cholesterol and TroponinT have the most missing values i.e. not measured for all patients. TroponinI is an indicator of heart muscle weakness
- 3 patients (RecordID 140501.0, 140936.0, 141264.0) don't have any time point measurements recorded
- 554 out of 4000 patients in seta have death recorded (~14%). This is a moderately imbalanced dataset with survival being the majority class (explains the overfitting of the baseline RF model)
    - The degree of imbalance is based on the following generally accepted criteria: 24-40% = mild, 1-20% - moderate, <1 - Extreme

## Background on features
Overview of physiological measurements, to gauge their utility as-is for the model versus need for feature engineering to make it usable or clear valid reason to discard it
- GCS = Glasgow Coma Scale
    - The Glasgow Coma Scale is a tool that healthcare providers use to measure decreases in consciousness. The scores from each section of the scale are useful for describing disruptions in nervous system function and also help providers track changes. It’s the most widely used tool for measuring comas and decreases in consciousness [source: Cleaveland Clinic](https://my.clevelandclinic.org/health/diagnostics/24848-glasgow-coma-scale-gcs)
    - Known interpretation (potential feature engineering target)
        - Severe <= 8
        - Moderate 9-12
        - Minor >= 13

- NIMAP = Non-Invasive Mean Arterial Blood Pressure 
    - Blood pressure measurement, most commonly through blood pressure cuffs
    - Known interpretation
        - Normal 70 - 100 mmHg
- Arterial pH
    - normal 7,35

# Hindsight 20:20 - what would I do differently if I started all over again
- Downloading the whole dataset was a waste of time and space. Next time, I would download just the set-a and outcome files

# What would I have done differently if I had more time?