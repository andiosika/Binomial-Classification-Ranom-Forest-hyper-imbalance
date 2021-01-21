## Project: Binomal Classification using Random Forest for Hyper-Imbalanced Target

Using binomial classification to predict COVID-19 infection on a large dataset (>618K samples) with extreme imbalance and minority class (.13% of samples) as target. 

The final iteration is a manually tuned random forsest classifier with >95% accuracy and >64% recall that uses biological, behavioral and environmental data collected to predict those who would test postive for COVID-19. Alternative methods were trialed using a Decision Tree and XGBoost, but Random Forest showed best baseline results.

**Main Files:** 

* workflow.ipynb - Code, details and visualizations 
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



### Data Background Observation: 
> The data was provided by subjects from 173 countries.  It is noted that 87% of the data comes from the US.  The next top provider of data is Canada ~5% , followed by the United Kingdom ~2.3%:

![GitHub Logo](/imgs/output_18_1.png)

#### Target Class is highly imbalanced: 

> Out of the nearly 618,134 samples, 893 tested positive for COVID-19, or .14% After preprocessing and modeling, the target occurance was .13% as indicated in the image below:

This is an approximate ratio of 1:700

![GitHub Logo](/imgs/output_129_2.png)

## The dataset:

Because of the size of this dataset, pandas profiling was used to inform potential considerations for dataset selection and develop a strategy to manage preprocessing of a set this size.  You can see the 43 features in the image below, as well as how complete the data collection was for each.

![GitHub Logo](/imgs/output_30_1.png)


#### Inspecting correlations:

```python
df_cor = pd.DataFrame(df.corr()['covid19_positive'].sort_values(ascending=False))
df_cor
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



## Main Dataset:

**Columns dropped:**
 Seven features of the original 43 were dropped in prior processing, rationale is below:
* **Date** While the date the data was collected could have a bearing on whether or not someone tested postivie, it would not provide insight to biological, behavioral or geographical indicators.
* **Region** This was a feature that substantially lacked data in the inital collection with 93% of the values missing.

In addition the following columns were dropped with rationale below:
* **ip_accuracy** - This feature measures the accuracy of the IP location and is used in the data collection process rather than for predicting a medical condition.
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

A baseline model using a basic decision tree was implemented with SMOTING data to address the class imbalance.  While accuracy for this model was 99% - it proved to be overtrained with recall rates for true negatives as well at .99 due to the large amout of data supporting this. Most importantly, recall for the target (covid positive) was .07 and this was the class we were most interested in predicting.

Several attempts were made implementing different attempts to deal with the extreme imbalance since the tartget class was also the minority class. Testing demonstrated that tuning the model's weight class returned increased performance over random oversampling or undersampling.  Various models were tested including Decision Tree, Random Forest and XGBoost.  When GridSearch was applied, modeling performed poorly when compared to manual tuning.  In some cases, the size of the dataset  proved to be too computationally expensive to implement GridSearch and testing began to run over 20 hours.  Because manual tuning proved to be more efficient, GridSearch was abandoned.  The final iteration was a manually tuned random forsest classifier with >90% accuracy and >64% recall or a improvement of **57% increase in recall** from the baseline.

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
fn.evaluate_model(X_test, y_test, yh8, X_train, y_train, rf_clf8)
```
![GitHub Logo](/imgs/best_model_results.PNG)

###  Observations on manually tuned random forest:

The overtraining data issues were addressed via several iterations of tuning the modeling.  **This model has an overall accuracy average of .96 and weighted recall of .96**  this iteration has yieled the highest true positive rate and is highest rated in terms of overall performance.  Mean Cross Validation Score revealed a result of 95.96%.

The area under the curve demonstrates the reliability of the model is 87.1% which is a 34% increase over the baseline of .53 introduced in the inital model.

The most important factors are listed below:

```python
fn.df_import(rf_clf8,X_train,n=10)
```
![GitHub Logo](/imgs/feature_importance_best_model.PNG)

#### Decision Tree visualizations from Random Forest Model:

![GitHub Logo](/imgs/output_249_0.png)


## Conclusion:

This dataset was a random sample of over 618,000 individuals reporting biological, behavioral, and environmental factors as well as their COVID-19 status.  The medium used to collect the data is operated by a UK - based an open-source self-reported platform that's main focus is data analytics and is non-medical in nature.  The questionaire used to collect data has undergone several versions and several features collected during this sample are no longer being tracked.  A very small rate (.013%) reported testing positive providing a hyper-imbalanced dataset.  It should be noted that at this time there was a shortage of tests available in the United States as well as time taken to get results was up to two weeks in latency.

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
>4.  If taking prescription medication, it is recommended to discuss additional risk for infection with a physician.

    
 
