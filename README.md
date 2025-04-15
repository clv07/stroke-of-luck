# stroke-of-luck

## Project Aim
12SL is the algorithm used by GE Healthcare Physicians in determining Myocardial Infarction (MI), or heart attack in a more common term from patients' electrocardiogram (ECG). We are training a machine learning model to assist physicians in quicker and more accurate clinical decisions making, especially under an emergency conditions. Our main goal from the model training is to reduce false negatives, where 12SL fails to detect MI when there's MI determined by physicians.

## Machine Learning Model

### Data Preprocessing
We read in data files provided by our mentor and ECG header files from [Physionet Challenge](). We preprocess the data files to join the tables into one big file, renaming and handling missing values, then separate into two big files which zooomed in about the data that we are interested about, specifically false positives and false negative data. 

### Model Design
We used lightweight gradient boosting (LGB) to train on false positive and false negative data respectively. 


---
We utilize ensemble model to stack different machine model to train the data.


### Performance
The model performance has 90% accuracy in F-1 score. There is x% of false negatives rate, it is a great improvement comparing to the initial performance of x%. 

[Graph]

Analysis

[Graph]

Analysis


### Sample Report


### Limitation
Our model has the limitation. 

## Credit
[Physionet Challenge](https://moody-challenge.physionet.org/2021/)
[Data source](https://github.com/physionetchallenges/python-classifier-2021)