## Project: Binomal Classification using Random Forest for Hyper-Imbalanced Target

Using binomial classification to predict COVID-19 infection on a large dataset (>618K samples) with extreme imbalance and minority class (.14% of samples) as target. 

The final iteration is a manually tuned random forsest classifier with >95% accuracy and >64% recall that uses biological, behavioral and environmental data collected to predict those who would test postive for COVID-19.

**Main Files:** 

* student.ipynb - Code, details and visualizations 
* Covid19Preso.pptx - A non-technical presentation with findings


* Blog post URL: https://andiosika.github.io/imbalanced_data


## Project Sections Within Main student.ipynb File:
**Link** | **Description**
--| --|
[Background](#Background:) | Details around the subject, datasource and objective
[Features and Descriptions](#Features-and-Descriptions:) | Details on each feature collected in the dataset
[Preprocessing](#Preprocessing:) | Steps taken to prepare data for modeling and evaluation
[Main Dataset](#Main-Dataset:) | The dataset in it's final form used for the predictive modeling results described in the [Conclusion](#Conclusion:)  section
[Modeling](#Modeling:) | Various iterations of predictive classification modeling including Decision Trees, Random Forest and XGBoost
[Best Model](#BEST-MODEL:-Manually-Tuned-Random-Forest) |Random Forest Classification Model including [Visualizations]() Confusion Matrix, ROC Curve, Feature Importance by Rank, Correlations
[Conclusion](#Conclusion:) | Summation of outcomes from modeling
    


<img src='https://github.com/andiosika/Binomial-Classification-Ranom-Forest-hyper-imbalance/blob/master/imgs/c0481846-wuhan_novel_coronavirus_illustration-spl.jpg' width=40% alignment=l>

____
## Background:

Coronavirus disease (COVID-19) is an infectious disease.  It was discovered in late 2019 and early 2020 and originated from Wuhan, China.  It escalated into a global pandemic.

According to the [World Health Organization](https://www.who.int/health-topics/coronavirus#tab=tab_1), most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.

The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. In response, much data has been collected in various ways to further inform ways to slow the spread. 

The dataset used in this evaluation was created by a project created by a UK based platform-solutions company called [Nexoid]( https://www.nexoid.com/). At the start of the pandemic, Nexoid noted that there was a lack of large datasets required to predict the spread and mortality rates related to COVID-19. They took it upon themselves to create and share this dataset as an effort to better understand these factors. It is a not-for-profit project with the goal of providing researchers and governments the data needed to help understand and fight COVID-19. It is a sample provided by self-reporting of over 618,000 individuals and collected a total of 43 biometric, behavioral, and environmental factors as well as their COVID-19 status.

The data is collected here: 
https://www.covid19survivalcalculator.com/ .  In exchange for the data, a risk of infection and mortality are returned to the user based on Nexoid's model which is not publicly sharded, yet recorded in this dataset post-hoc.  These values are reflected in the columns risk_infection and risk_mortality.

The questionaire used to collect data has since undergone several versions and several features collected during this sample are no longer being tracked. Data for this observation was collected between March 27 - April 10 of 2020, and only a very small rate (.13%) of respondents reported testing postive for COVID-19. It should be noted that at this time there was a shortage of tests available in the United States and latency in recieving results was up to two weeks. 


**The intention of this classification project is to identify primary contributing factors for contracting COVID-19.**

## The features data was collected on are detailed in the table below:

**Feature** | **Description**
--|--
survey_date|The date the survey was submitted
region	
country |The country collected from IP address long, lat
ip_latitude	|ip latitude of device at time of survey
ip_longitude |ip longitude of device at time of survey	
ip_accuracy	|-n/a
sex	|Self reported sex
age	| Self reported age based on birthdate
height |Height in cm
weight | Weight in kg
bmi	| Body Mass Index as calculated from self-reported height and weight
blood_type	| Blood type
smoking	| reported smoking/vapeing habits (never, do, 1-5x, 6-20x, 20+, quit<5yrs, quit>5yrs, quit>10yrs
alcohol	| reported days of alcohol consuption in last 14 days 
cannabis | reported days of cannabis consumpiton in last 28 days
amphetamines | reported days of amphetamine consumpiton in last 28 days	
cocaine	| reported days of cocaine consumpiton in last 28 days
lsd	| reported days of lsd consumpiton in last 28 days
mdma | reported days of mdma(ecstacy) consumpiton in last 28 days	
contacts_count	| reported contacts in the last week (1-20 and 20+)
house_count	| how many people live in the subjects dwelling
text_working | work/school travel behaviors (0-5 never did, always did, have stopped, critical only, still do)
rate_government_action	| scale of attitude that government is taking covid-19 seriously (disagree, neutral, agree)
rate_reducing_risk_single | scale of self-assesment to reduce risk(social distancing, hand washing) (disagree, neutral, agree)
rate_reducing_risk_house | scale of assessesed co-habitators risk reduction (social distancing, hand washing)(disagree, neutral, agree)	
rate_reducing_mask	| scale of how often a mask is worn outside dwelling 1-5 rarely, sometimes, usually)
covid19_positive | A binomial value o=no, 1=yes to the question  "Do you have?"	
covid19_symptoms | A binomial value o=no, 1=yes to the question  "Do you have?"	
covid19_contact	|A binomial value 0=no, 1=yes to the question "Have you been in contact with someone who has tested positive?"
asthma | A binomial value 0=no, 1=yes to the question "Do you have?"
kidney_disease | A binomial value 0=no, 1=yes to the question "Do you have?"
compromised_immune |  A binomial value 0=no, 1=yes to the question "Do you have?"
heart_disease | A binomial value 0=no, 1=yes to the question "Do you have?"	
lung_disease | A binomial value 0=no, 1=yes to the question "Do you have?"
diabetes | A binomial value 0=no, 1=yes to the question "Do you have?"
hiv_positive | A binomial value 0=no, 1=yes to the question "Do you have?"
hypertension | A binomial value 0=no, 1=yes to the question "Do you have?"
other_chronic | A binomial value 0=no, 1=yes to the question "Do you have?"
prescription_medication | Reported prescription medications
opinion_infection | No information is given about this feature, no longer collecting data on this, it is theorized that it had to do with if the subject believed they had the infection.
opinion_mortality | No information is given about this feature, no longer collecting data on this, it is theorized that it had to do with if the subject believed they could die from the infection.
risk_infection | calc'd risk for infection (based on their models)
risk_mortality | calc'd risk for mortality (based on their models)
   
___


### Data Background Observation: 
> The data was provided by subjects from 173 countries.  It is noted that 87% of the data comes from the US.  The next top provider of data is Canada ~5% , followed by the United Kingdom ~2.3%:

![GitHub Logo](/imgs/output_18_1.png)

#### Target Class is highly imbalanced: 

> Out of the nearly 618,134 samples, 893 tested positive for COVID-19, or .14%

This is an approximate ratio of 1:700

![GitHub Logo](/imgs/output_129_2.png)

#### Inspecting correlations:


```python
df_cor = pd.DataFrame(df.corr()['covid19_positive'].sort_values(ascending=False))
df_cor
```

```python
df.corr()['covid19_positive'].sort_values(ascending=False).plot(kind='barh', figsize=(12,12))
```

### Raw Data Inspection Observations: 

A quick look revealed that were no direct correlations and the most highly correlated features of the unprocessed data are: 

Feature: |  Correlation: 
 --| --|
covid19_positive |	1.000000
risk_infection | 0.198632
covid19_symptoms | 0.089861
opinion_infection |	0.054837
covid19_contact | 0.050774
risk_mortality | 0.014074
mdma | 0.012152
heart_disease | 0.007975
weight | 0.007503
lsd |	0.007137
height | 0.006999


## Preprocessing:

This section outlines steps taken to prepare the data for analysis. The first step was to address missing/null values.  

Because of the size of this dataset, pandas profiling was used to inform potential considerations for dataset selection and develop a strategy to manage preprocessing of a set this size.

![GitHub Logo](/imgs/output_30_1.png)

Initial visual inspection of null values indicates that region and prescription medication are sparsely populated.  Since region was ~90% missing, it was dropped.  Prescription medication had 57K values and details are [included in this section](#Prescription-Medication). 

The opinion_infections and opinion_mortality columns are also a little 'light' in terms of responses and have the same number of responses.  This null rate of ~16% was imputed with the median values for each respective field. 

Null values in columns that contain <5% null values were dropped.  

Other than those outlined above, there doesn't seem to be be any other apparent patterns for incomplete data. (See above).


## Main Dataset:

**Columns dropped:**
 These columns were dropped in prior processing:
>* **Date** While the date the data was collected could have a bearing on whether or not someone tested postivie, it would not provide insight to biological, behavioral or geographical indicators.
* **Region** This was a feature that substantially lacked data in the inital collection with 93% of the values missing.

In addition the following columns were dropped with rationale below:
>* **ip_accuracy** - This feature measures the accuracy of the IP location and is used in the data collection process rather than for predicting a medical condition.
* **risk_infection** - This is a value calculated post-hoc, based on the data collected from this dataset
* **risk_mortality** - This is a value calculated post-hoc, based on the data collected from this dataset
* **prescription_medication** - This column contains text-strings and has over 57K values. A column was added called **taking_prescription_medication** to capture if an individual is taking prescribed medicine.  It's proposed to deal with this column separately if it's indicated to be a factor separately since this is computationally expensive.


## Train/Test Split:

```python
from sklearn.model_selection import train_test_split
```

```python
y = df2['covid19_positive'].copy()
X = df2.drop('covid19_positive', axis=1).copy()

```

```python
y.value_counts()
```

    0    574419
    1       791
    Name: covid19_positive, dtype: int64




```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, stratify=y, random_state=123)
```


```python
len(X_test)
```




    143803




```python
len(y_test)
```




    143803




```python
len(y_train)
```




    431407




```python
len(X_train)
```




    431407




```python
print(df['covid19_positive'].value_counts(normalize=True))
coviddf= pd.DataFrame(df['covid19_positive'].value_counts(normalize=True)*100)
coviddf.plot(kind='bar', color='r')
plt.title('Covid19 Positive Rates')

```

    0    0.998625
    1    0.001375
    Name: covid19_positive, dtype: float64
    




    Text(0.5, 1.0, 'Covid19 Positive Rates')


### Inspecting training set  for imbalance 


```python
y_train.value_counts()
```




    0    430814
    1       593
    Name: covid19_positive, dtype: int64




```python
y_test.value_counts()
```




    0    143605
    1       198
    Name: covid19_positive, dtype: int64



## Modeling:  

First Attempt: **Using 'vanilla' Decision Tree and SMOTE to address imbalances**

Several attempts were made implementing different attempts to deal with the extreme imbalance since the tartget class was also the minority class. Testing demonstrated that tuning the model's weight class returned increased performance over random oversampling or undersampling.  Various models were tested including Decision Tree, Random Forest and XGBoost.  When GridSearch was applied, modeling performed poorly when compared to manual tuning.  In some cases, the size of the dataset  proved to be too computationally expensive to implement GridSearch and testing began to run over 20 hours.  Because manual tuning proved to be more efficient, GridSearch was abandoned.  The final iteration was a manually tuned random forsest classifier with >90% accuracy and >64% recall.






## BEST MODEL: Manually tuned Random Forest





```python
time = fn.Timer()
time.start()
rf_clf8 = RandomForestClassifier(criterion='gini', max_depth=2, max_features=.45, class_weight='balanced',n_estimators=80, random_state=111)
rf_clf8.fit(X_train, y_train)
time.stop()
```


```python
yh8=rf_clf8.predict(X_test)
```


```python
mean_rf_cv_score = np.mean(cross_val_score(rf_clf8, X_train, y_train, cv=3))

print(f"Mean Cross Validation Score for Random Forest Classifier: {mean_rf_cv_score :.2%}")
```


```python
fn.evaluate_model(X_test, y_test, yh8, X_train, y_train, rf_clf8)
```

###  Observations on manually tuned random forest:

The overtraining data issues were addressed via several iterations of tuning the modeling.  **This model has an overall accuracy average of .96 and weighted recall of .96**  this iteration has yieled the highest true positive rate and is highest rated in terms of overall performance.  Mean Cross Validation Score revealed a result of 95.96%.

The area under the curve demonstrates the reliability of the model is 87.1% which is a 34% increase over the baseline of .53 introduced in the inital model.

The most important factors are listed below:



```python
fn.df_import(rf_clf8,X_train,n=10)
```


```python
fn.plot_importance(rf_clf8,X_train)

```

#### Decision Tree visualizations from Random Forest Model:


```python
dot_data1 = export_graphviz(rf_clf8.estimators_[3], out_file=None, 
                           feature_names=X_train.columns,  
                           class_names=np.unique(y).astype('str'), 
                           filled=True, rounded=True, special_characters=True)

# Draw graph
graph1 = graph_from_dot_data(dot_data1)  

# Show graph
Image(graph1.create_png())
```

#### Attempting Randomized Search:

**WARNING: The following 7 input lines tast take 94.5 mins to run and have been commented out.** 

The model was clearly overtrained and performed poorly.  Observations are recorded below:


```python
from sklearn.model_selection import RandomizedSearchCV
```


```python
# stop
# time = fn.Timer()
# time.start()
# rf_clfb = RandomForestClassifier(class_weight='balanced', random_state=111)
# ## Set up param grid
# param_grid = {'criterion':['gini','entropy'],
#              'max_depth':[7,8, 10,15],
#              'max_features':[.2, .3, .45],
#              'n_estimators' :[75,100,125, 150]}

# ## Instantiate GridSearchCV
# rgrid_clfb = RandomizedSearchCV(rf_clfb, param_grid, n_jobs=-1, verbose=1, cv=skf)
# time.stop()
```


```python
#rgrid_clfb.fit(X_train, y_train)
```


```python
yhtrgrid = rgrid_clfb.predict(X_test)
```


```python
rgrid_clfb.best_params_
```


```python
rf_clfb1 = RandomForestClassifier(criterion = 'gini', n_estimators=100, max_features=.2, 
                                  max_depth=15, class_weight='balanced', random_state=111)
time = fn.Timer()
time.start()
rf_clfb1.fit(X_train, y_train)
time.stop()
```


```python
# hytb1 = rf_clfb1.predict(X_test)
```


```python
#fn.evaluate_model(X_test, y_test, hytb1, X_train_res, y_train_res, rf_clfb1)
```

#### Observations:
Validates that a manually tuned Random Forest model performed best.  Depite the AUC remaining relatively high(86.1), the true positive rate is extremely poor at .12.

precision | recall |	f1-score |	support
--| --| --| --|
0	|**0.999**	|0.998|	0.999|	143605.000
1	|0.085|	**0.116**	|0.098|	198.000
accuracy	|0.997	|0.997|	0.997	|0.997
macro avg|	0.542|	0.557	|0.548	|143803.000
weighted avg|	0.998|0.997|	0.997	|143803.000
___________________________________________


Training Accuracy :  0.9988614576127981
Test Accuracy :  0.9970584758315195
____________________________________

## Conclusion:

This dataset was a random sample of over 618,000 individuals reporting biological, behavioral, and environmental factors as well as their COVID-19 status.  The medium used to collect the data is operated by a UK - based an open-source platform that's main focus is data analytics and is non-medical in nature.  The questionaire used to collect data has undergone several versions and several features collected during this sample are no longer being tracked.  A very small rate (.013%) reported testing positive providing a hyper-imbalanced dataset.  It should be noted that at this time there was a shortage of tests available in the United States as well as time taken to get results was up to two weeks in latency.

Using data collected over a 15 day span (March 27 - April 10, 2020), a predictive model was developed to identify top factors in contracting COVID-19.

The factors that rated highest in predicting contraction of COVID-19 were derived using a Random Forest Classification model that yeilded an overall accuracy average of .97 and weighted recall of .97.  A Receiver Operator Characteristic (ROC) curve demonstrates a diagnostic ability of this binary classifier to be 84.9%.  This model ran at a higher sensitivy rate of 96% than it's specificity rate of 64%. 

The most important factors are listed below.  The + or - signs indicate whether the correlations associated with these factors were positive or negative:

Factor| Description |	Importance | Correlation
--| --| --| --|
opinion_infection| Individual believed they contracted COVID-19| 	0.529735 | +
covid19_symptoms| Individual exhibited COVID-19 symptops| 	0.285415 | +
covid19_contact | Individual came in contact with another who was COVID-19+ |	0.122519 | +
rate_reducing_risk_house | Househould practiced social distancing and hygiene |	0.0136553 | -
omwasnull | Individual did not respond if they believed they could die from COVID-19| 0.0127974 | +
rate_reducing_risk_single |Individual practiced social distancing and hygiene |	0.0108048 | -
oiwasnull | Individual did not respond if they believed they could die from COVID-19|0.00731367|+
sex_male | Indivdiual was male |	0.00641232 | +
sex_female | Individual was female |	0.00354491 | -
bmi | Body Mass Index (kg/m** 2)|	0.00247806 | +
taking prescription medication | The individual was taking prescription medication | 0.001504 | +





It's common sense that having symptoms and coming into contact with someone infected would be factors in contracting a contagious disease, and are proven as such since they classified as top factors. More information is needed on  'opinion_infection' as this is also a top factor.  It is believed that this feature indicates the individual is infected with COVID-19. However there is no background and data is no longer being collected on this datapoint. 

The fields 'oiwasnull' and 'omwasnull' is associated with this factor - and reflects that respondents did not fill this value in.  Possible suggested reasons for this are 1) lack of attention to detail by the respondant, 2) stigma or other psychological rationale associated with being COVID-19+ 3) Survey stopped collecting data.  

This model also illustrates that behavioral modifications such as taking precautionary measures by practicing social distancing and hygiene individually and as a household as important factors in predicting disease contraction.  It should be noted that collective action ranked higher than individual action and the more these behaviors increased, classifing the individual as COVID-19+ decreased.  In addition, these two factors were the two most negatively correlated factors with testing positive.

Sex ranked in the top 10 as important factors as well. Men ranked higher than women and this is accentuated by the fact that men had a positive correlation with becoming infected and women had a negative correlation.   

Both BMI and taking prescriptions even relatively small in comparison were identified in terms of feature importance.  
There are many possible reasons why increased BMI could be associated with an increased succeptibility.  [This study](https://watermark.silverchair.com/ciaa415.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAoQwggKABgkqhkiG9w0BBwagggJxMIICbQIBADCCAmYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMVXV_JGiK8L-3DL_MAgEQgIICN77iaVCmYvPz7CwP7dpJRua4RrrP384UJAH-QXvKyiAQrwDWcADuTGgZMbhF_qxtAG8LP6XC4F70O1TyJx6B5w896GRNhvMIRaKWdWNm3JeCVlzXVX9LLVgAshV2CIL8UZSmLmUwBIMU0v8_8tKA_QloitlBiu_TvBmea4YaA99nNj-IrQjleAwriZjRHKIeaI3EFJuXYMuaBuOFGBMReNvDq1gfYxwmrbC4aX0W7sYRHPE7YFsfb4AZDMsVdx8-2t7Sn9VSmK0jP-dq46OmhC8Ed_MfuyAhRaqTGClD5OAyNSuZYF_ErgEf0OEX5dcXyMBJLnPlVi71thCX_MjcQBmhvJdymnIYXRc_PJUSj1N0V1uKZe4YBTfU00sN2Uke-UCqHo_34C7axYyYoW2wfcw3fLb-VtWMITOSM4oZkPu7oGX5Mq5IH0jr4d3oUKQD5Ar_lD6yG2p5F7c-yczARNpL109KjfBEd760prsGbft8Cd9GGKkGuR8bnpY0q1QMKuVHE1BCXqrwRa2Ypjx8oKL0Cor3ZyPPBGJbDBUBe9oMf8BbFwRqUa02fIQ66gt7-lHzvPZ2_1aLmQBvCmWmWq8OEkkkaem0_kVLDobAXTfhMwV-Ho7AT_5V9S1UOY0TCjn-I0_FvKKrEqi9qMBz_gJhUabfjd6Ph9YZpCiQO2z9YpyufH-OE-4Un7CSSLAsyHnpir5V55iJjiDsv0HA1ULdYcnE2Pkkj8EWjaUrAsqTbq0Lwb2z-g) supports those who have a bmi greater than 35 are at higher risk.  Future work could be done to further investigate the BMI feature interpret if this dataset aligns with this model.

Those taking prescription medications could suggest a state poor health, but claiming this rationale is somewhat presumptive and would obviously need further investigation.  The data provided did include quantities and labels for each prescription medication, however since the nature of 57000 unique values in the original dataset, it was tracked to evaluate feature importance.  Future work could be done in this area while taking into consideration the ranking of it's relatively low feature importance. 

## Recommendations: 
>1. If an individual has symptoms associated with COVID-19, or think they could be infected, testing is recommended for confirmation.
>2. Practice social distancing and hygiene individually and as a household
>       *This is especially true for males and those with high BMI.
>3. Avoid contact with those known to be infected.
?4.  If taking prescription medication, it is recommended to discuss additional risk for infection with a physician.

    
 



## Appendix: 

This section contains code for some of the visualizations and supporting information used in the non-technial presentation.


```python
importlib.reload(fn)
```


```python

```


```python
#getting all necessary feature imporance values from best model
corrs = pd.Series(rf_clf8.feature_importances_, index=X_train.columns, name='importance')
x = corrs.sort_values(ascending=False).head(11)
x
```


```python
#df with all features and corresponding correlation values
df_cor = pd.DataFrame(df2.corr()['covid19_positive'].sort_values(ascending=False))
df_cor
```


```python
#pulling all correlations for each of the important features for best model:
y = df_cor.loc[list(x.index),:]
y
```

### Vizualizations for non-technical presentation:


```python
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


labels = ['COVID-19 -','COVID-19 +']

cnf_matrix = metrics.plot_confusion_matrix(rf_clf8,X_test,y_test,cmap='Reds',
                              normalize='true',display_labels=labels)
plt.title('Random Forest Classifier Prediction Rates', fontdict=font)


```


```python
pos_map = {'0' : 'Covid19 Negative',
          '1': 'Covid19 Posistive'}

coviddf['covid19_positive'] = coviddf['covid19_positive'].map(pos_map)
```


```python
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 18,
        }

font1 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
print(df['covid19_positive'].value_counts(normalize=True))
lables = ['Negative', 'Positive']
coviddf= pd.DataFrame(df['covid19_positive'].value_counts(normalize=True)*100)
coviddf.plot(kind='bar', color='darkred')
plt.title('COVID-19 Rates', fontdict=font)
plt.xlabel('Negative                :::               Positive', fontdict=font1)

plt.ylabel('Percent', fontdict=font1)



```


```python
fig= plt.figure()
df_import = pd.Series(rf_clf8.feature_importances_, index=X_train.columns, name='importance')
df_import.sort_values().tail(11).plot(kind='barh',color='red', figsize=(7,5))
plt.title('Top 10 Feature Importance for the Contraction of COVID-19')




```


```python
import pyplot
```


```python
df['covid19_positive'].value_counts()
```


```python
#!pip install Counter
from collections import Counter
# summarize class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
data = {"x":[], "y":[], "label":[]}
for label, coord in counter.items():
    data["x"].append(coord[0])
    data["y"].append(coord[1])
    label["label"].append(label)
    
plt.figure(figsize=(10,8))
plt.title('Scatter Plot', fontsize=20)
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.scatter(data["x"], data["y"], marker = 'o')

# add labels
for label, x, y in zip(data["label"], data["x"], data["y"]):
    plt.annotate(label, xy = (x, y))
```


```python
counter.items()
```


```python
# featimpt = ['covid19_symptoms','opinion_infection', 'covid19_contact','rate_reducing_risk_house',
# 'taking_prescription_medication','text_working_travel critical','rate_reducing_risk_single',
# 'smoking_never', 'rate_reducing_mask', 'oiwasnull']featimpt
```


```python
importlib.reload(fn)
```


```python
df_cor.loc['smoking':]
```


```python
df_cor.head(60)
```


```python
df_cor.tail(50)
```


```python
def _plot_classification_report(y_true, y_pred_class):
    import sklearn.metrics as metrics
    report = metrics.classification_report(y_true, y_pred_class, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=report_df.values,
             colLabels=report_df.columns,
             rowLabels=report_df.index,
             loc='center',
             bbox=[0.2, 0.2, 0.8, 0.8])
    fig.tight_layout()

    return fig 
```


```python
#alternative code for feature importance:
#df_import_tree = pd.Series(tree.feature_importances_, index=X_train.columns, name='importance').head(20)
# df_import_tree.sort_values().plot(kind='barh', figsize=(15,12))
```


```python
!pip install mictools
```


```python
!pip install ppscore
```


```python

```
