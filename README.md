# Stroke of Luck 
[Repository Link](https://github.com/clv07/stroke-of-luck)

## Project Overview
12SL is the algorithm used by GE Healthcare Physicians in determining Myocardial Infarction (MI), or heart attack in a more common term from patients' electrocardiogram (ECG). We are training a machine learning model to assist physicians in quicker and more accurate clinical decisions making, especially under emergency conditions. Our main goal from the model training is to reduce false negatives, where 12SL fails to detect MI when there's MI determined by physicians.

## Installation and Setup
1. Clone the repository.
```
git clone https://github.com/clv07/stroke-of-luck.git
```
2. Setup and activate a virtual environment.
```
python3 -m venv venv
```
3. Install required libraries. 
```
pip install -r requirements.txt
```
4. Run the script.
```
python3 mi_detection.py
```

## Code Structure

### Data Preprocessing
We first read in the ECG data provided by our mentors along with header files from the [Physionet Challenge](https://moody-challenge.physionet.org/2021/). The data preprocessing step involves merging multiple tables into a single consolidated file, standardizing column names, handling missing values, and then splitting the dataset into false positive and false negative groups for separate model training.

### Model Overview
Our model is designed as a two-pass ensemble system that improves upon the 12SL algorithm for detecting myocardial infarction (MI) from ECG data, with a particular focus on reducing false negatives and improving diagnostic precision.

In the first pass, the model classifies ECG signals as either Positive (likely MI) or Negative (likely Non-MI, NMI) categories. We trained two separate Light Gradient Boosting Machine (LightGBM) classifiers. One model focuses on false positive cases, while the other model focuses on false negative cases.

These two models handle misclassifications from the 12SL outputs and provide improved initial categorization.

In the second pass, we introduce a Severity model, also based on LightGBM, to further assess Positive cases and predict whether an MI case is Acute or Non-Acute. The Severity model focuses on acute MI detection, highlighting the difficulty of fine-grained severity classification.

Overall, the architecture prioritizes interpretability and precision. The Positive or Negative classifiers emphasize broader MI detection. The Severity model targets clinical urgency (Acute vs. Non-Acute MI), assisting physicians in rapid triage decisions.

This two-pass ensemble approach enables better correction of 12SL’s misclassifications and provides an additional layer of clinical decision support for emergency interventions.

![Machine Learning Model Design](readme_images/models.png)

### Example Report Output
After running the model, we generate a final prediction report based on the sample ECG file. 
The report contains:
1. Predicted MI Status: Indicates whether the model classifies the ECG as Positive (MI) or Negative (No MI).
2. Predicted Severity: For Positive cases, further classification into Acute MI or Non-Acute MI.
3. Confidence Scores: Probability estimates from the LightGBM classifiers for both MI detection and severity prediction.

![Example Report Output]()

## Achievements
Our model achieved strong performance in improving myocardial infarction detection. Specifically:
- Positive classifier achieved an F1 score of 0.95.
- Negative classifier achieved an F1 score of 0.65.
- Severity classifier achieved an F1 score of 0.33.

The Positive classifier significantly reduced false negative rates compared to the baseline 12SL algorithm, enhancing the model’s reliability in emergency clinical settings.

![Confusion Matrix]()

![Decision Tree Plot]()

![PCA Graph]()

![SHAP Graph]()

## Limitations

While our model significantly improves upon the baseline 12SL algorithm, there are several limitations that must be considered. 

First, the Severity model for predicting Acute vs. Non-Acute MI achieved a relatively low F1 score (0.33), suggesting that finer-grained classification remains challenging. Additionally, model performance may vary across different patient demographics and clinical settings, as the training data was limited to a specific dataset. 

Another limitation is that some ECG features may be underutilized or not fully captured, which could affect the model's generalization capabilities. Finally, external validation on independent datasets has not yet been conducted, and would be necessary to confirm the model's generalizability before clinical deployment.


## Future Work

To further improve our model and its clinical applicability, several future directions are proposed. First, enhancing the Severity model remains a priority. By incorporating additional features or exploring alternative modeling approaches, such as deep learning models that better capture temporal ECG patterns, we can aim to improve fine-grained severity classification.

Another important area is expanding the dataset. Including more diverse and representative patient samples would strengthen the model’s robustness and improve its generalizability across different populations and clinical settings.

Furthermore, conducting real-world clinical validations is crucial. Prospective studies and live clinical trials are needed to rigorously assess the model's effectiveness and safety before potential deployment.

Improving model interpretability is another key focus. In addition to SHAP, exploring methods like counterfactual explanations could help make the model’s predictions more transparent and actionable for clinicians.

Lastly, integration with hospital information systems should be investigated to enable seamless deployment of our model within clinical workflows, ultimately supporting faster and more reliable MI diagnosis in real-world settings.