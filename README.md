## Mod 3 Project: Binomal Classification

Using binomial classification to predict COVID-19 infection on a large dataset (>618K samples) with extreme imbalance and minority class (.13% of samples) as target. 

The final iteration is a manually tuned random forsest classifier with >90% accuracy and >64% recall.

**Main Files:** 

* student.ipynb - Code, details and visualizations 
* Covid19Preso.pptx - A non-technical presentation with findings


* Blog post URL: https://andiosika.github.io/imbalanced_data


## Project Quick Links:
**Link** | **Description**
--| --|
[Background](#Background:) | Details around the subject, datasource and objective
[Features and Descriptions](#Features-and-Descriptions:) | Details on each feature collected in the dataset
[Preprocessing](#Preprocessing:) | Steps taken to prepare data for modeling and evaluation
[Main Dataset](#Main-Dataset:) | The dataset in it's final form used for the predictive modeling results described in the [Conclusion](#Conclusion:)  section
[Modeling](#Modeling:) | Various iterations of predictive classification modeling including Decision Trees, Random Forest and XGBoost
[Best Model](#BEST-MODEL:-Manually-Tuned-Random-Forest) |Random Forest Classification Model including [Visualizations]() Confusion Matrix, ROC Curve, Feature Importance by Rank, Correlations
    [Conclusion](#Conclusion:) | Summation of outcomes from modeling
    


<img src='https://raw.githubusercontent.com/andiosika/dsc-mod-3-project-v2-1-online-ds-pt-100719/master/c0481846-wuhan_novel_coronavirus_illustration-spl.jpg' width=40% alignment=l>

## Background:

Coronavirus disease (COVID-19) is an infectious disease.  It was discovered in late 2019 and early 2020 and originated from Wuhan, China.  It escalated into a global pandemic.

According to the [World Health Organization](https://www.who.int/health-topics/coronavirus#tab=tab_1), most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.

The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. In response, much data has been collected in various ways to further inform ways to slow the spread. 

The dataset used in this evaluation was created by a project created by a UK based platform-solutions company called [Nexoid]( https://www.nexoid.com/). At the start of the pandemic, Nexoid noted that there was a lack of large datasets required to predict the spread and mortality rates related to COVID-19. They took it upon themselves to create and share this dataset as an effort to better understand these factors. It is a not-for-profit project with the goal of providing researchers and governments the data needed to help understand and fight COVID-19. It is a sample provided by self-reporting of over 618,000 individuals and collects biological, behavioral, and environmental factors as well as their COVID-19 status.

The data is collected here: 
https://www.covid19survivalcalculator.com/ .  In exchange for the data, a risk of infection and mortality are returned to the user based on Nexoid's model which is not publicly sharded, yet recorded in this dataset post-hoc.  These values are reflected in the columns risk_infection and risk_mortality.

The questionaire used to collect data has since undergone several versions and several features collected during this sample are no longer being tracked. Data for this observation was collected between March 27 - April 10 of 2020, and only a very small rate (.13%) of respondents reported testing postive for COVID-19. It should be noted that at this time there was a shortage of tests available in the United States and latency in recieving results was up to two weeks. 


**The intention of this classification project is to identify primary contributing factors for contracting COVID-19.**


```python
##Importing dataset
import pandas as pd
df = pd.read_csv("master_dataset4.csv")
pd.set_option('display.max_columns', 0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>survey_date</th>
      <th>region</th>
      <th>country</th>
      <th>ip_latitude</th>
      <th>ip_longitude</th>
      <th>ip_accuracy</th>
      <th>sex</th>
      <th>age</th>
      <th>height</th>
      <th>weight</th>
      <th>bmi</th>
      <th>blood_type</th>
      <th>smoking</th>
      <th>alcohol</th>
      <th>cannabis</th>
      <th>amphetamines</th>
      <th>cocaine</th>
      <th>lsd</th>
      <th>mdma</th>
      <th>contacts_count</th>
      <th>house_count</th>
      <th>text_working</th>
      <th>rate_government_action</th>
      <th>rate_reducing_risk_single</th>
      <th>rate_reducing_risk_house</th>
      <th>rate_reducing_mask</th>
      <th>covid19_positive</th>
      <th>covid19_symptoms</th>
      <th>covid19_contact</th>
      <th>asthma</th>
      <th>kidney_disease</th>
      <th>compromised_immune</th>
      <th>heart_disease</th>
      <th>lung_disease</th>
      <th>diabetes</th>
      <th>hiv_positive</th>
      <th>hypertension</th>
      <th>other_chronic</th>
      <th>prescription_medication</th>
      <th>opinion_infection</th>
      <th>opinion_mortality</th>
      <th>risk_infection</th>
      <th>risk_mortality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/4/2020</td>
      <td>NaN</td>
      <td>US</td>
      <td>35.9568</td>
      <td>-86.5301</td>
      <td>200.0</td>
      <td>female</td>
      <td>40_50</td>
      <td>158</td>
      <td>114</td>
      <td>45.6</td>
      <td>ap</td>
      <td>quit0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>5.0</td>
      <td>4</td>
      <td>never</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>35.0</td>
      <td>64.248</td>
      <td>0.721</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/5/2020</td>
      <td>NaN</td>
      <td>US</td>
      <td>39.6512</td>
      <td>-82.6200</td>
      <td>20.0</td>
      <td>female</td>
      <td>20_30</td>
      <td>168</td>
      <td>62</td>
      <td>21.9</td>
      <td>on</td>
      <td>never</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>5.0</td>
      <td>3</td>
      <td>never</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>57.549</td>
      <td>0.016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4/5/2020</td>
      <td>NaN</td>
      <td>US</td>
      <td>27.7723</td>
      <td>-82.2767</td>
      <td>10.0</td>
      <td>female</td>
      <td>0_10</td>
      <td>136</td>
      <td>44</td>
      <td>23.7</td>
      <td>bp</td>
      <td>never</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>stopped</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>62.377</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4/5/2020</td>
      <td>NaN</td>
      <td>US</td>
      <td>39.6675</td>
      <td>-77.5666</td>
      <td>10.0</td>
      <td>female</td>
      <td>30_40</td>
      <td>164</td>
      <td>112</td>
      <td>41.6</td>
      <td>abn</td>
      <td>quit5</td>
      <td>-1.0</td>
      <td>28.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>never</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>ACETAZOLAMIDE;GABAPENTIN;OMEPRAZOLE;VENLAFAXIN...</td>
      <td>45.0</td>
      <td>25.0</td>
      <td>59.258</td>
      <td>0.195</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4/4/2020</td>
      <td>NaN</td>
      <td>US</td>
      <td>41.3527</td>
      <td>-81.7444</td>
      <td>50.0</td>
      <td>male</td>
      <td>50_60</td>
      <td>184</td>
      <td>132</td>
      <td>38.9</td>
      <td>an</td>
      <td>vape</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>-1.0</td>
      <td>10.0</td>
      <td>4</td>
      <td>travel critical</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>METFORMIN HYDROCHLORIDE</td>
      <td>15.0</td>
      <td>5.0</td>
      <td>77.098</td>
      <td>2.463</td>
    </tr>
  </tbody>
</table>
</div>



### Features and Descriptions:

There are 43 features on which data was collected around biometetrics, behavior and enviromnent.  

Details are below:

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


## Inspecting the dataset:

#### Software Package Installs:


```python
# Package Installs
import matplotlib.pyplot as plt

import seaborn as sns
from pandas_profiling import ProfileReport
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import functions as fn
import importlib

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
```

    C:\Users\aosika\AppData\Local\Continuum\anaconda3\envs\learn-env\lib\site-packages\sklearn\externals\six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    

    2020-05-07 09:49:31.767234-07:00
    [i] Timer started at05/07/20 - 09:49 AM
    [i] Timer ended at 05/07/20 - 09:49 AM
    - Total time = 0:00:00
    

    C:\Users\aosika\AppData\Local\Continuum\anaconda3\envs\learn-env\lib\site-packages\sklearn\utils\deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)
    

This set of data contains just over 619K entries and has 43 columns of both numeric and categorical data.  Because of the size of this dataset, pandas profiling was used to inform potential considerations for dataset selection and develop a strategy to manage preprocessing of a set this size.

### Data Background Observation: 
> The data was provided by subjects from 173 countries.  It is noted that 87% of the data comes from the US.  The next top provider of data is Canada ~5% , followed by the United Kingdom ~2.3%:


```python
countriesdf.head(5).plot(kind='bar', color='r')
plt.title('US Represents 87% of Data:')
```




    Text(0.5, 1.0, 'US Represents 87% of Data:')




![png](output_14_1.png)



```python
df['covid19_positive'].value_counts()
```




    0    618134
    1       893
    Name: covid19_positive, dtype: int64



#### Target Class is highly imbalanced: 

> Out of the nearly 618,134 samples, 893 tested positive for COVID-19, or .0014%

This is an approximate ratio of 1:1000

#### Inspecting correlations:


```python
df_cor = pd.DataFrame(df.corr()['covid19_positive'].sort_values(ascending=False))
df_cor
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>covid19_positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>covid19_positive</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>risk_infection</th>
      <td>0.198632</td>
    </tr>
    <tr>
      <th>covid19_symptoms</th>
      <td>0.089861</td>
    </tr>
    <tr>
      <th>opinion_infection</th>
      <td>0.054837</td>
    </tr>
    <tr>
      <th>covid19_contact</th>
      <td>0.050774</td>
    </tr>
    <tr>
      <th>risk_mortality</th>
      <td>0.014074</td>
    </tr>
    <tr>
      <th>mdma</th>
      <td>0.012152</td>
    </tr>
    <tr>
      <th>heart_disease</th>
      <td>0.007975</td>
    </tr>
    <tr>
      <th>weight</th>
      <td>0.007503</td>
    </tr>
    <tr>
      <th>lsd</th>
      <td>0.007137</td>
    </tr>
    <tr>
      <th>height</th>
      <td>0.006999</td>
    </tr>
    <tr>
      <th>cocaine</th>
      <td>0.006833</td>
    </tr>
    <tr>
      <th>rate_reducing_mask</th>
      <td>0.006201</td>
    </tr>
    <tr>
      <th>ip_longitude</th>
      <td>0.006122</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>0.005700</td>
    </tr>
    <tr>
      <th>kidney_disease</th>
      <td>0.004725</td>
    </tr>
    <tr>
      <th>other_chronic</th>
      <td>0.004638</td>
    </tr>
    <tr>
      <th>compromised_immune</th>
      <td>0.004308</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>0.004280</td>
    </tr>
    <tr>
      <th>hypertension</th>
      <td>0.004055</td>
    </tr>
    <tr>
      <th>hiv_positive</th>
      <td>0.003993</td>
    </tr>
    <tr>
      <th>contacts_count</th>
      <td>0.003741</td>
    </tr>
    <tr>
      <th>ip_latitude</th>
      <td>0.003448</td>
    </tr>
    <tr>
      <th>lung_disease</th>
      <td>0.003296</td>
    </tr>
    <tr>
      <th>amphetamines</th>
      <td>0.002425</td>
    </tr>
    <tr>
      <th>asthma</th>
      <td>0.001956</td>
    </tr>
    <tr>
      <th>house_count</th>
      <td>-0.001151</td>
    </tr>
    <tr>
      <th>ip_accuracy</th>
      <td>-0.001347</td>
    </tr>
    <tr>
      <th>opinion_mortality</th>
      <td>-0.002450</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>-0.004070</td>
    </tr>
    <tr>
      <th>cannabis</th>
      <td>-0.004418</td>
    </tr>
    <tr>
      <th>rate_government_action</th>
      <td>-0.005191</td>
    </tr>
    <tr>
      <th>rate_reducing_risk_house</th>
      <td>-0.010192</td>
    </tr>
    <tr>
      <th>rate_reducing_risk_single</th>
      <td>-0.013982</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.corr()['covid19_positive'].sort_values(ascending=False).plot(kind='barh', figsize=(12,12))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2278095ccc0>




![png](output_19_1.png)



```python
df.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1) 
```




<style  type="text/css" >
    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col0 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col1 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col2 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col3 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col4 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col5 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col6 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col7 {
            background-color:  #adc9fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col8 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col9 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col10 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col11 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col12 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col13 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col14 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col15 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col16 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col17 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col18 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col19 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col20 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col21 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col22 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col23 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col24 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col25 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col26 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col27 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col28 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col29 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col30 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col31 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col32 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col33 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col1 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col2 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col3 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col4 {
            background-color:  #9abbff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col5 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col6 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col7 {
            background-color:  #98b9ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col8 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col9 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col10 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col11 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col12 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col13 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col14 {
            background-color:  #b5cdfa;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col15 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col16 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col17 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col18 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col19 {
            background-color:  #a6c4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col20 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col21 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col22 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col23 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col24 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col25 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col26 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col27 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col28 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col29 {
            background-color:  #a5c3fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col30 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col31 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col32 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col33 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col0 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col1 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col2 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col3 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col4 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col5 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col6 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col7 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col8 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col9 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col10 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col11 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col12 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col13 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col14 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col16 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col17 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col18 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col19 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col20 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col21 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col22 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col23 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col24 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col25 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col26 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col27 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col28 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col29 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col30 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col31 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col32 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col33 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col0 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col1 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col2 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col3 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col4 {
            background-color:  #ccd9ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col6 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col7 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col8 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col9 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col10 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col11 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col12 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col13 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col14 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col15 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col16 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col17 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col18 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col19 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col20 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col21 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col22 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col23 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col24 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col25 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col26 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col27 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col28 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col29 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col30 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col31 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col32 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col33 {
            background-color:  #6f92f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col0 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col1 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col2 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col3 {
            background-color:  #c7d7f0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col4 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col5 {
            background-color:  #dc5d4a;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col7 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col8 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col9 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col10 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col11 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col12 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col13 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col14 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col15 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col16 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col17 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col18 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col19 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col20 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col21 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col22 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col23 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col24 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col25 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col26 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col27 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col28 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col29 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col30 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col31 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col32 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col33 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col0 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col1 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col2 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col3 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col4 {
            background-color:  #da5a49;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col5 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col7 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col8 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col9 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col10 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col11 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col12 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col13 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col14 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col15 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col16 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col17 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col18 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col19 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col20 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col21 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col22 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col23 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col24 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col25 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col26 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col27 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col28 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col29 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col30 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col31 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col32 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col33 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col0 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col1 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col2 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col3 {
            background-color:  #80a3fa;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col4 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col5 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col6 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col7 {
            background-color:  #7396f5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col8 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col9 {
            background-color:  #80a3fa;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col10 {
            background-color:  #7a9df8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col11 {
            background-color:  #7da0f9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col12 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col13 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col14 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col15 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col16 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col17 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col18 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col19 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col20 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col21 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col22 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col23 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col24 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col25 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col26 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col27 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col28 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col29 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col30 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col31 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col32 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col33 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col0 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col1 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col2 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col3 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col4 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col5 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col6 {
            background-color:  #6a8bef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col7 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col8 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col9 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col10 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col11 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col12 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col13 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col15 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col16 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col17 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col18 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col19 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col20 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col21 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col22 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col23 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col24 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col25 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col26 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col27 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col28 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col29 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col30 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col31 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col32 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col33 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col0 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col1 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col2 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col3 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col4 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col5 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col6 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col7 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col8 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col9 {
            background-color:  #81a4fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col10 {
            background-color:  #85a8fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col11 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col12 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col13 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col14 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col16 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col17 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col18 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col19 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col20 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col21 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col22 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col23 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col24 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col25 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col26 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col27 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col28 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col29 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col30 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col31 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col32 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col33 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col0 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col1 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col2 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col3 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col4 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col5 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col6 {
            background-color:  #6f92f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col7 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col8 {
            background-color:  #86a9fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col9 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col10 {
            background-color:  #d5dbe5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col11 {
            background-color:  #dedcdb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col12 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col13 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col14 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col16 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col17 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col18 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col19 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col20 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col21 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col22 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col23 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col24 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col25 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col26 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col27 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col28 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col29 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col30 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col31 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col32 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col33 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col0 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col1 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col2 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col3 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col4 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col5 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col6 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col7 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col8 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col9 {
            background-color:  #d4dbe6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col10 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col11 {
            background-color:  #f7b99e;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col12 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col13 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col15 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col16 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col17 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col18 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col19 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col20 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col21 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col22 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col23 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col24 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col25 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col26 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col27 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col28 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col29 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col30 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col31 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col32 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col33 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col0 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col1 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col2 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col3 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col4 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col5 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col6 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col7 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col8 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col9 {
            background-color:  #dcdddd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col10 {
            background-color:  #f7b99e;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col11 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col12 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col13 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col14 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col16 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col17 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col18 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col19 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col20 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col21 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col22 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col23 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col24 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col25 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col26 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col27 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col28 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col29 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col30 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col31 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col32 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col33 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col0 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col1 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col2 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col3 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col4 {
            background-color:  #7597f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col5 {
            background-color:  #7597f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col6 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col7 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col8 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col9 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col10 {
            background-color:  #6a8bef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col11 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col12 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col13 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col14 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col16 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col17 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col18 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col19 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col20 {
            background-color:  #9abbff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col21 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col22 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col23 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col24 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col25 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col26 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col27 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col28 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col29 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col30 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col31 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col32 {
            background-color:  #f7b79b;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col33 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col0 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col1 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col2 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col3 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col4 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col5 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col6 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col7 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col8 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col9 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col10 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col11 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col12 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col13 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col14 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col15 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col16 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col17 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col18 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col19 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col20 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col21 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col22 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col23 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col24 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col25 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col26 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col27 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col28 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col29 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col30 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col31 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col32 {
            background-color:  #bcd2f7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col33 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col0 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col1 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col2 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col3 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col4 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col5 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col6 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col7 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col8 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col9 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col10 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col11 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col12 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col13 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col14 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col15 {
            background-color:  #7396f5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col16 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col17 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col18 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col19 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col20 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col21 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col22 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col23 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col24 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col25 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col26 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col27 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col28 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col29 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col30 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col31 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col32 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col33 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col0 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col1 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col2 {
            background-color:  #98b9ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col3 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col4 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col5 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col6 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col7 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col8 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col9 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col10 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col11 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col12 {
            background-color:  #7b9ff9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col13 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col14 {
            background-color:  #b7cff9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col15 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col16 {
            background-color:  #f7af91;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col17 {
            background-color:  #d6dce4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col18 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col19 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col20 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col21 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col22 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col23 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col24 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col25 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col26 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col27 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col28 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col29 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col30 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col31 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col32 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col33 {
            background-color:  #98b9ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col0 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col1 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col2 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col3 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col4 {
            background-color:  #93b5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col5 {
            background-color:  #8fb1fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col6 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col7 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col8 {
            background-color:  #8badfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col9 {
            background-color:  #86a9fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col10 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col11 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col12 {
            background-color:  #779af7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col13 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col14 {
            background-color:  #afcafc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col15 {
            background-color:  #f7b599;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col16 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col17 {
            background-color:  #c5d6f2;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col18 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col19 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col20 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col21 {
            background-color:  #8fb1fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col22 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col23 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col24 {
            background-color:  #8fb1fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col25 {
            background-color:  #8fb1fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col26 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col27 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col28 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col29 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col30 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col31 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col32 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col33 {
            background-color:  #92b4fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col0 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col1 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col2 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col3 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col4 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col5 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col6 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col7 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col8 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col9 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col10 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col11 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col12 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col13 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col14 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col15 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col16 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col17 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col18 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col19 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col20 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col21 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col22 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col23 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col24 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col25 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col26 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col27 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col28 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col29 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col30 {
            background-color:  #7396f5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col31 {
            background-color:  #84a7fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col32 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col33 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col0 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col1 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col2 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col3 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col4 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col5 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col6 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col7 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col8 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col9 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col10 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col11 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col12 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col13 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col14 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col16 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col17 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col18 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col19 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col20 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col21 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col22 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col23 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col24 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col25 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col26 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col27 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col28 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col29 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col30 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col31 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col32 {
            background-color:  #7ea1fa;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col33 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col0 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col1 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col2 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col3 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col4 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col5 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col6 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col7 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col8 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col9 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col10 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col11 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col12 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col13 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col14 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col15 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col16 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col17 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col18 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col19 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col20 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col21 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col22 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col23 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col24 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col25 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col26 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col27 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col28 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col29 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col30 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col31 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col32 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col33 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col0 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col1 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col2 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col3 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col4 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col5 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col6 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col7 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col8 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col9 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col10 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col11 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col12 {
            background-color:  #7a9df8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col13 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col15 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col16 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col17 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col18 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col19 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col20 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col21 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col22 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col23 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col24 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col25 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col26 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col27 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col28 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col29 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col30 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col31 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col32 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col33 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col0 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col1 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col2 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col4 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col5 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col6 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col7 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col8 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col9 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col10 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col11 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col12 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col13 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col14 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col15 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col16 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col17 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col18 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col19 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col20 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col21 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col22 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col23 {
            background-color:  #6a8bef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col24 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col25 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col26 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col27 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col28 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col29 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col30 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col31 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col32 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col33 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col0 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col1 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col2 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col3 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col4 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col5 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col7 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col8 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col9 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col10 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col11 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col12 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col13 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col14 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col16 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col17 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col18 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col19 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col20 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col21 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col22 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col23 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col24 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col25 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col26 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col27 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col28 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col29 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col30 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col31 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col32 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col33 {
            background-color:  #7396f5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col0 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col1 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col2 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col4 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col5 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col6 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col7 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col8 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col9 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col10 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col11 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col12 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col13 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col14 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col15 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col16 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col17 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col18 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col19 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col20 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col21 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col22 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col23 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col24 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col25 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col26 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col27 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col28 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col29 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col30 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col31 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col32 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col33 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col0 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col1 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col2 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col3 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col4 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col5 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col6 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col7 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col8 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col9 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col10 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col11 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col12 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col13 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col14 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col16 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col17 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col18 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col19 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col20 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col21 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col22 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col23 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col24 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col25 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col26 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col27 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col28 {
            background-color:  #7093f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col29 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col30 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col31 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col32 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col33 {
            background-color:  #bfd3f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col0 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col1 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col2 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col3 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col4 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col5 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col6 {
            background-color:  #3c4ec2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col7 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col8 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col9 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col10 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col11 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col12 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col13 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col14 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col16 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col17 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col18 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col19 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col20 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col21 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col22 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col23 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col24 {
            background-color:  #6b8df0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col25 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col26 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col27 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col28 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col29 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col30 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col31 {
            background-color:  #7093f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col32 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col33 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col0 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col1 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col2 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col3 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col4 {
            background-color:  #84a7fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col5 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col7 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col8 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col9 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col10 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col11 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col12 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col13 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col14 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col15 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col16 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col17 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col18 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col19 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col20 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col21 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col22 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col23 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col24 {
            background-color:  #799cf8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col25 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col26 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col27 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col28 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col29 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col30 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col31 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col32 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col33 {
            background-color:  #94b6ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col0 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col1 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col2 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col3 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col4 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col5 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col6 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col7 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col8 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col9 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col10 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col11 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col12 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col13 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col15 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col16 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col17 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col18 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col19 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col20 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col21 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col22 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col23 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col24 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col25 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col26 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col27 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col28 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col29 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col30 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col31 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col32 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col33 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col0 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col1 {
            background-color:  #3e51c5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col2 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col3 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col4 {
            background-color:  #8fb1fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col5 {
            background-color:  #89acfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col6 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col7 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col8 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col9 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col10 {
            background-color:  #445acc;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col11 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col12 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col13 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col14 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col15 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col16 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col17 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col18 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col19 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col20 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col21 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col22 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col23 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col24 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col25 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col26 {
            background-color:  #8caffe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col27 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col28 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col29 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col30 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col31 {
            background-color:  #7295f4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col32 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col33 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col0 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col1 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col2 {
            background-color:  #465ecf;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col4 {
            background-color:  #4f69d9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col5 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col6 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col7 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col8 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col9 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col10 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col11 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col12 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col13 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col15 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col16 {
            background-color:  #4358cb;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col17 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col18 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col19 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col20 {
            background-color:  #4b64d5;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col21 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col22 {
            background-color:  #5673e0;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col23 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col24 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col25 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col26 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col27 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col28 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col29 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col30 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col31 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col32 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col33 {
            background-color:  #6687ed;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col0 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col1 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col2 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col3 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col4 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col5 {
            background-color:  #6180e9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col6 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col7 {
            background-color:  #6282ea;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col8 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col9 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col10 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col11 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col12 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col13 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col14 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col15 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col16 {
            background-color:  #4c66d6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col17 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col18 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col19 {
            background-color:  #779af7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col20 {
            background-color:  #8badfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col21 {
            background-color:  #6c8ff1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col22 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col23 {
            background-color:  #688aef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col24 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col25 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col26 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col27 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col28 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col29 {
            background-color:  #6a8bef;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col30 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col31 {
            background-color:  #a3c2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col32 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col33 {
            background-color:  #455cce;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col0 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col1 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col2 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col3 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col4 {
            background-color:  #7597f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col5 {
            background-color:  #85a8fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col6 {
            background-color:  #3f53c6;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col7 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col8 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col9 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col10 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col11 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col12 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col13 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col14 {
            background-color:  #3d50c3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col15 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col16 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col17 {
            background-color:  #7b9ff9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col18 {
            background-color:  #5470de;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col19 {
            background-color:  #5572df;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col20 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col21 {
            background-color:  #93b5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col22 {
            background-color:  #6485ec;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col23 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col24 {
            background-color:  #7699f6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col25 {
            background-color:  #85a8fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col26 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col27 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col28 {
            background-color:  #81a4fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col29 {
            background-color:  #8db0fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col30 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col31 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col32 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col33 {
            background-color:  #84a7fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col0 {
            background-color:  #a7c5fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col1 {
            background-color:  #97b8ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col2 {
            background-color:  #b9d0f9;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col3 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col4 {
            background-color:  #b1cbfc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col5 {
            background-color:  #aec9fc;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col6 {
            background-color:  #9bbcff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col7 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col8 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col9 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col10 {
            background-color:  #a9c6fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col11 {
            background-color:  #aac7fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col12 {
            background-color:  #f59f80;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col13 {
            background-color:  #e6d7cf;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col14 {
            background-color:  #98b9ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col15 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col16 {
            background-color:  #4a63d3;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col17 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col18 {
            background-color:  #cedaeb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col19 {
            background-color:  #c4d5f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col20 {
            background-color:  #d5dbe5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col21 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col22 {
            background-color:  #a2c1ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col23 {
            background-color:  #9abbff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col24 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col25 {
            background-color:  #9dbdff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col26 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col27 {
            background-color:  #a1c0ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col28 {
            background-color:  #9fbfff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col29 {
            background-color:  #98b9ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col30 {
            background-color:  #cedaeb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col31 {
            background-color:  #96b7ff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col32 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col33 {
            background-color:  #90b2fe;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col0 {
            background-color:  #536edd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col1 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col2 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col3 {
            background-color:  #6e90f2;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col4 {
            background-color:  #6788ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col5 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col6 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col7 {
            background-color:  #485fd1;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col8 {
            background-color:  #5977e3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col9 {
            background-color:  #5e7de7;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col10 {
            background-color:  #5f7fe8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col11 {
            background-color:  #5b7ae5;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col12 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col13 {
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col14 {
            background-color:  #5a78e4;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col15 {
            background-color:  #4961d2;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col16 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col17 {
            background-color:  #6384eb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col18 {
            background-color:  #5875e1;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col19 {
            background-color:  #516ddb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col20 {
            background-color:  #4e68d8;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col21 {
            background-color:  #506bda;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col22 {
            background-color:  #88abfd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col23 {
            background-color:  #7093f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col24 {
            background-color:  #cbd8ee;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col25 {
            background-color:  #9ebeff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col26 {
            background-color:  #9abbff;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col27 {
            background-color:  #5d7ce6;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col28 {
            background-color:  #abc8fd;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col29 {
            background-color:  #7093f3;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col30 {
            background-color:  #4257c9;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col31 {
            background-color:  #82a6fb;
            color:  #000000;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col32 {
            background-color:  #4055c8;
            color:  #f1f1f1;
        }    #T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col33 {
            background-color:  #b40426;
            color:  #f1f1f1;
        }</style><table id="T_fced5690_9082_11ea_88ae_54e1adf13a74" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >ip_latitude</th>        <th class="col_heading level0 col1" >ip_longitude</th>        <th class="col_heading level0 col2" >ip_accuracy</th>        <th class="col_heading level0 col3" >height</th>        <th class="col_heading level0 col4" >weight</th>        <th class="col_heading level0 col5" >bmi</th>        <th class="col_heading level0 col6" >alcohol</th>        <th class="col_heading level0 col7" >cannabis</th>        <th class="col_heading level0 col8" >amphetamines</th>        <th class="col_heading level0 col9" >cocaine</th>        <th class="col_heading level0 col10" >lsd</th>        <th class="col_heading level0 col11" >mdma</th>        <th class="col_heading level0 col12" >contacts_count</th>        <th class="col_heading level0 col13" >house_count</th>        <th class="col_heading level0 col14" >rate_government_action</th>        <th class="col_heading level0 col15" >rate_reducing_risk_single</th>        <th class="col_heading level0 col16" >rate_reducing_risk_house</th>        <th class="col_heading level0 col17" >rate_reducing_mask</th>        <th class="col_heading level0 col18" >covid19_positive</th>        <th class="col_heading level0 col19" >covid19_symptoms</th>        <th class="col_heading level0 col20" >covid19_contact</th>        <th class="col_heading level0 col21" >asthma</th>        <th class="col_heading level0 col22" >kidney_disease</th>        <th class="col_heading level0 col23" >compromised_immune</th>        <th class="col_heading level0 col24" >heart_disease</th>        <th class="col_heading level0 col25" >lung_disease</th>        <th class="col_heading level0 col26" >diabetes</th>        <th class="col_heading level0 col27" >hiv_positive</th>        <th class="col_heading level0 col28" >hypertension</th>        <th class="col_heading level0 col29" >other_chronic</th>        <th class="col_heading level0 col30" >opinion_infection</th>        <th class="col_heading level0 col31" >opinion_mortality</th>        <th class="col_heading level0 col32" >risk_infection</th>        <th class="col_heading level0 col33" >risk_mortality</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row0" class="row_heading level0 row0" >ip_latitude</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col0" class="data row0 col0" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col1" class="data row0 col1" >-0.47</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col2" class="data row0 col2" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col3" class="data row0 col3" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col4" class="data row0 col4" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col5" class="data row0 col5" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col6" class="data row0 col6" >-0.0015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col7" class="data row0 col7" >0.034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col8" class="data row0 col8" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col9" class="data row0 col9" >0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col10" class="data row0 col10" >-0.0053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col11" class="data row0 col11" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col12" class="data row0 col12" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col13" class="data row0 col13" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col14" class="data row0 col14" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col15" class="data row0 col15" >0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col16" class="data row0 col16" >0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col17" class="data row0 col17" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col18" class="data row0 col18" >0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col19" class="data row0 col19" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col20" class="data row0 col20" >0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col21" class="data row0 col21" >0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col22" class="data row0 col22" >-0.00053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col23" class="data row0 col23" >0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col24" class="data row0 col24" >-0.0045</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col25" class="data row0 col25" >0.0018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col26" class="data row0 col26" >0.0016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col27" class="data row0 col27" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col28" class="data row0 col28" >-0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col29" class="data row0 col29" >0.0042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col30" class="data row0 col30" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col31" class="data row0 col31" >0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col32" class="data row0 col32" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row0_col33" class="data row0 col33" >-0.004</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row1" class="row_heading level0 row1" >ip_longitude</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col0" class="data row1 col0" >-0.47</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col1" class="data row1 col1" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col2" class="data row1 col2" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col3" class="data row1 col3" >0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col4" class="data row1 col4" >-0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col5" class="data row1 col5" >-0.058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col6" class="data row1 col6" >-0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col7" class="data row1 col7" >-0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col8" class="data row1 col8" >-0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col9" class="data row1 col9" >-0.00027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col10" class="data row1 col10" >-0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col11" class="data row1 col11" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col12" class="data row1 col12" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col13" class="data row1 col13" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col14" class="data row1 col14" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col15" class="data row1 col15" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col16" class="data row1 col16" >0.0086</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col17" class="data row1 col17" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col18" class="data row1 col18" >0.0061</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col19" class="data row1 col19" >0.0088</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col20" class="data row1 col20" >-0.0084</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col21" class="data row1 col21" >-0.001</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col22" class="data row1 col22" >-0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col23" class="data row1 col23" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col24" class="data row1 col24" >-5.9e-06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col25" class="data row1 col25" >-0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col26" class="data row1 col26" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col27" class="data row1 col27" >0.00038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col28" class="data row1 col28" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col29" class="data row1 col29" >0.00099</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col30" class="data row1 col30" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col31" class="data row1 col31" >-0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col32" class="data row1 col32" >-0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row1_col33" class="data row1 col33" >-0.012</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row2" class="row_heading level0 row2" >ip_accuracy</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col0" class="data row2 col0" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col1" class="data row2 col1" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col2" class="data row2 col2" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col3" class="data row2 col3" >0.0081</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col4" class="data row2 col4" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col5" class="data row2 col5" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col6" class="data row2 col6" >-0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col7" class="data row2 col7" >0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col8" class="data row2 col8" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col9" class="data row2 col9" >0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col10" class="data row2 col10" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col11" class="data row2 col11" >0.0022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col12" class="data row2 col12" >0.098</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col13" class="data row2 col13" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col14" class="data row2 col14" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col15" class="data row2 col15" >-0.034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col16" class="data row2 col16" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col17" class="data row2 col17" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col18" class="data row2 col18" >-0.0013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col19" class="data row2 col19" >-0.003</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col20" class="data row2 col20" >0.007</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col21" class="data row2 col21" >-0.0089</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col22" class="data row2 col22" >0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col23" class="data row2 col23" >-0.003</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col24" class="data row2 col24" >0.00049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col25" class="data row2 col25" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col26" class="data row2 col26" >-0.00014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col27" class="data row2 col27" >-0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col28" class="data row2 col28" >0.0021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col29" class="data row2 col29" >-0.0082</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col30" class="data row2 col30" >0.00022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col31" class="data row2 col31" >-0.00086</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col32" class="data row2 col32" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row2_col33" class="data row2 col33" >-0.015</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row3" class="row_heading level0 row3" >height</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col0" class="data row3 col0" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col1" class="data row3 col1" >0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col2" class="data row3 col2" >0.0081</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col3" class="data row3 col3" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col4" class="data row3 col4" >0.39</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col5" class="data row3 col5" >-0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col6" class="data row3 col6" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col7" class="data row3 col7" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col8" class="data row3 col8" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col9" class="data row3 col9" >0.043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col10" class="data row3 col10" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col11" class="data row3 col11" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col12" class="data row3 col12" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col13" class="data row3 col13" >-0.06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col14" class="data row3 col14" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col15" class="data row3 col15" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col16" class="data row3 col16" >0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col17" class="data row3 col17" >-0.046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col18" class="data row3 col18" >0.007</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col19" class="data row3 col19" >-0.0065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col20" class="data row3 col20" >-0.0055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col21" class="data row3 col21" >-0.067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col22" class="data row3 col22" >0.0015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col23" class="data row3 col23" >-0.057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col24" class="data row3 col24" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col25" class="data row3 col25" >-0.0065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col26" class="data row3 col26" >0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col27" class="data row3 col27" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col28" class="data row3 col28" >0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col29" class="data row3 col29" >-0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col30" class="data row3 col30" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col31" class="data row3 col31" >-0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col32" class="data row3 col32" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row3_col33" class="data row3 col33" >0.085</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row4" class="row_heading level0 row4" >weight</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col0" class="data row4 col0" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col1" class="data row4 col1" >-0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col2" class="data row4 col2" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col3" class="data row4 col3" >0.39</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col4" class="data row4 col4" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col5" class="data row4 col5" >0.87</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col6" class="data row4 col6" >-0.066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col7" class="data row4 col7" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col8" class="data row4 col8" >-0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col9" class="data row4 col9" >-0.0032</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col10" class="data row4 col10" >-0.0009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col11" class="data row4 col11" >-0.0059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col12" class="data row4 col12" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col13" class="data row4 col13" >0.00021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col14" class="data row4 col14" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col15" class="data row4 col15" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col16" class="data row4 col16" >-0.0029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col17" class="data row4 col17" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col18" class="data row4 col18" >0.0075</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col19" class="data row4 col19" >0.0036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col20" class="data row4 col20" >-0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col21" class="data row4 col21" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col22" class="data row4 col22" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col23" class="data row4 col23" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col24" class="data row4 col24" >0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col25" class="data row4 col25" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col26" class="data row4 col26" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col27" class="data row4 col27" >0.0094</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col28" class="data row4 col28" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col29" class="data row4 col29" >0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col30" class="data row4 col30" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col31" class="data row4 col31" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col32" class="data row4 col32" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row4_col33" class="data row4 col33" >0.065</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row5" class="row_heading level0 row5" >bmi</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col0" class="data row5 col0" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col1" class="data row5 col1" >-0.058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col2" class="data row5 col2" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col3" class="data row5 col3" >-0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col4" class="data row5 col4" >0.87</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col5" class="data row5 col5" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col6" class="data row5 col6" >-0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col7" class="data row5 col7" >-0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col8" class="data row5 col8" >-0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col9" class="data row5 col9" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col10" class="data row5 col10" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col11" class="data row5 col11" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col12" class="data row5 col12" >0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col13" class="data row5 col13" >0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col14" class="data row5 col14" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col15" class="data row5 col15" >-0.0085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col16" class="data row5 col16" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col17" class="data row5 col17" >-0.0073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col18" class="data row5 col18" >0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col19" class="data row5 col19" >0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col20" class="data row5 col20" >-0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col21" class="data row5 col21" >0.085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col22" class="data row5 col22" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col23" class="data row5 col23" >0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col24" class="data row5 col24" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col25" class="data row5 col25" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col26" class="data row5 col26" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col27" class="data row5 col27" >-0.0066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col28" class="data row5 col28" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col29" class="data row5 col29" >0.049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col30" class="data row5 col30" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col31" class="data row5 col31" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col32" class="data row5 col32" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row5_col33" class="data row5 col33" >0.027</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row6" class="row_heading level0 row6" >alcohol</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col0" class="data row6 col0" >-0.0015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col1" class="data row6 col1" >-0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col2" class="data row6 col2" >-0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col3" class="data row6 col3" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col4" class="data row6 col4" >-0.066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col5" class="data row6 col5" >-0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col6" class="data row6 col6" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col7" class="data row6 col7" >0.076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col8" class="data row6 col8" >0.037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col9" class="data row6 col9" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col10" class="data row6 col10" >0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col11" class="data row6 col11" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col12" class="data row6 col12" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col13" class="data row6 col13" >-0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col14" class="data row6 col14" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col15" class="data row6 col15" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col16" class="data row6 col16" >0.0059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col17" class="data row6 col17" >-0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col18" class="data row6 col18" >-0.0041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col19" class="data row6 col19" >-0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col20" class="data row6 col20" >0.0058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col21" class="data row6 col21" >-0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col22" class="data row6 col22" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col23" class="data row6 col23" >-0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col24" class="data row6 col24" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col25" class="data row6 col25" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col26" class="data row6 col26" >-0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col27" class="data row6 col27" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col28" class="data row6 col28" >-0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col29" class="data row6 col29" >-0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col30" class="data row6 col30" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col31" class="data row6 col31" >-0.076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col32" class="data row6 col32" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row6_col33" class="data row6 col33" >0.024</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row7" class="row_heading level0 row7" >cannabis</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col0" class="data row7 col0" >0.034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col1" class="data row7 col1" >-0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col2" class="data row7 col2" >0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col3" class="data row7 col3" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col4" class="data row7 col4" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col5" class="data row7 col5" >-0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col6" class="data row7 col6" >0.076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col7" class="data row7 col7" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col8" class="data row7 col8" >0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col9" class="data row7 col9" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col10" class="data row7 col10" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col11" class="data row7 col11" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col12" class="data row7 col12" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col13" class="data row7 col13" >-0.0073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col14" class="data row7 col14" >-0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col15" class="data row7 col15" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col16" class="data row7 col16" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col17" class="data row7 col17" >-0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col18" class="data row7 col18" >-0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col19" class="data row7 col19" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col20" class="data row7 col20" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col21" class="data row7 col21" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col22" class="data row7 col22" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col23" class="data row7 col23" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col24" class="data row7 col24" >-0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col25" class="data row7 col25" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col26" class="data row7 col26" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col27" class="data row7 col27" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col28" class="data row7 col28" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col29" class="data row7 col29" >0.03</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col30" class="data row7 col30" >0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col31" class="data row7 col31" >0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col32" class="data row7 col32" >-0.0053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row7_col33" class="data row7 col33" >-0.043</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row8" class="row_heading level0 row8" >amphetamines</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col0" class="data row8 col0" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col1" class="data row8 col1" >-0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col2" class="data row8 col2" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col3" class="data row8 col3" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col4" class="data row8 col4" >-0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col5" class="data row8 col5" >-0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col6" class="data row8 col6" >0.037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col7" class="data row8 col7" >0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col8" class="data row8 col8" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col9" class="data row8 col9" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col10" class="data row8 col10" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col11" class="data row8 col11" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col12" class="data row8 col12" >0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col13" class="data row8 col13" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col14" class="data row8 col14" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col15" class="data row8 col15" >-0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col16" class="data row8 col16" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col17" class="data row8 col17" >-0.0091</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col18" class="data row8 col18" >0.0024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col19" class="data row8 col19" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col20" class="data row8 col20" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col21" class="data row8 col21" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col22" class="data row8 col22" >0.0095</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col23" class="data row8 col23" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col24" class="data row8 col24" >0.0085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col25" class="data row8 col25" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col26" class="data row8 col26" >0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col27" class="data row8 col27" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col28" class="data row8 col28" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col29" class="data row8 col29" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col30" class="data row8 col30" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col31" class="data row8 col31" >0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col32" class="data row8 col32" >0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row8_col33" class="data row8 col33" >0.015</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row9" class="row_heading level0 row9" >cocaine</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col0" class="data row9 col0" >0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col1" class="data row9 col1" >-0.00027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col2" class="data row9 col2" >0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col3" class="data row9 col3" >0.043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col4" class="data row9 col4" >-0.0032</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col5" class="data row9 col5" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col6" class="data row9 col6" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col7" class="data row9 col7" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col8" class="data row9 col8" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col9" class="data row9 col9" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col10" class="data row9 col10" >0.44</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col11" class="data row9 col11" >0.47</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col12" class="data row9 col12" >0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col13" class="data row9 col13" >0.004</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col14" class="data row9 col14" >-0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col15" class="data row9 col15" >-0.063</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col16" class="data row9 col16" >-0.049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col17" class="data row9 col17" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col18" class="data row9 col18" >0.0068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col19" class="data row9 col19" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col20" class="data row9 col20" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col21" class="data row9 col21" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col22" class="data row9 col22" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col23" class="data row9 col23" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col24" class="data row9 col24" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col25" class="data row9 col25" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col26" class="data row9 col26" >0.0011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col27" class="data row9 col27" >0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col28" class="data row9 col28" >-0.0064</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col29" class="data row9 col29" >0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col30" class="data row9 col30" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col31" class="data row9 col31" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col32" class="data row9 col32" >0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row9_col33" class="data row9 col33" >0.033</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row10" class="row_heading level0 row10" >lsd</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col0" class="data row10 col0" >-0.0053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col1" class="data row10 col1" >-0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col2" class="data row10 col2" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col3" class="data row10 col3" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col4" class="data row10 col4" >-0.0009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col5" class="data row10 col5" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col6" class="data row10 col6" >0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col7" class="data row10 col7" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col8" class="data row10 col8" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col9" class="data row10 col9" >0.44</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col10" class="data row10 col10" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col11" class="data row10 col11" >0.64</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col12" class="data row10 col12" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col13" class="data row10 col13" >0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col14" class="data row10 col14" >-0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col15" class="data row10 col15" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col16" class="data row10 col16" >-0.042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col17" class="data row10 col17" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col18" class="data row10 col18" >0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col19" class="data row10 col19" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col20" class="data row10 col20" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col21" class="data row10 col21" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col22" class="data row10 col22" >0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col23" class="data row10 col23" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col24" class="data row10 col24" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col25" class="data row10 col25" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col26" class="data row10 col26" >0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col27" class="data row10 col27" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col28" class="data row10 col28" >-0.0038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col29" class="data row10 col29" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col30" class="data row10 col30" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col31" class="data row10 col31" >0.0097</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col32" class="data row10 col32" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row10_col33" class="data row10 col33" >0.036</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row11" class="row_heading level0 row11" >mdma</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col0" class="data row11 col0" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col1" class="data row11 col1" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col2" class="data row11 col2" >0.0022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col3" class="data row11 col3" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col4" class="data row11 col4" >-0.0059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col5" class="data row11 col5" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col6" class="data row11 col6" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col7" class="data row11 col7" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col8" class="data row11 col8" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col9" class="data row11 col9" >0.47</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col10" class="data row11 col10" >0.64</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col11" class="data row11 col11" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col12" class="data row11 col12" >0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col13" class="data row11 col13" >0.0075</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col14" class="data row11 col14" >-0.045</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col15" class="data row11 col15" >-0.054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col16" class="data row11 col16" >-0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col17" class="data row11 col17" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col18" class="data row11 col18" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col19" class="data row11 col19" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col20" class="data row11 col20" >0.0088</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col21" class="data row11 col21" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col22" class="data row11 col22" >0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col23" class="data row11 col23" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col24" class="data row11 col24" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col25" class="data row11 col25" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col26" class="data row11 col26" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col27" class="data row11 col27" >0.064</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col28" class="data row11 col28" >-0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col29" class="data row11 col29" >0.0028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col30" class="data row11 col30" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col31" class="data row11 col31" >-0.0032</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col32" class="data row11 col32" >0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row11_col33" class="data row11 col33" >0.024</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row12" class="row_heading level0 row12" >contacts_count</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col0" class="data row12 col0" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col1" class="data row12 col1" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col2" class="data row12 col2" >0.098</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col3" class="data row12 col3" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col4" class="data row12 col4" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col5" class="data row12 col5" >0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col6" class="data row12 col6" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col7" class="data row12 col7" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col8" class="data row12 col8" >0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col9" class="data row12 col9" >0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col10" class="data row12 col10" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col11" class="data row12 col11" >0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col12" class="data row12 col12" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col13" class="data row12 col13" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col14" class="data row12 col14" >-0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col15" class="data row12 col15" >-0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col16" class="data row12 col16" >-0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col17" class="data row12 col17" >-0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col18" class="data row12 col18" >0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col19" class="data row12 col19" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col20" class="data row12 col20" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col21" class="data row12 col21" >0.0011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col22" class="data row12 col22" >-0.0036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col23" class="data row12 col23" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col24" class="data row12 col24" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col25" class="data row12 col25" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col26" class="data row12 col26" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col27" class="data row12 col27" >-0.0055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col28" class="data row12 col28" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col29" class="data row12 col29" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col30" class="data row12 col30" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col31" class="data row12 col31" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col32" class="data row12 col32" >0.62</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row12_col33" class="data row12 col33" >-0.089</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row13" class="row_heading level0 row13" >house_count</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col0" class="data row13 col0" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col1" class="data row13 col1" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col2" class="data row13 col2" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col3" class="data row13 col3" >-0.06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col4" class="data row13 col4" >0.00021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col5" class="data row13 col5" >0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col6" class="data row13 col6" >-0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col7" class="data row13 col7" >-0.0073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col8" class="data row13 col8" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col9" class="data row13 col9" >0.004</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col10" class="data row13 col10" >0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col11" class="data row13 col11" >0.0075</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col12" class="data row13 col12" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col13" class="data row13 col13" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col14" class="data row13 col14" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col15" class="data row13 col15" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col16" class="data row13 col16" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col17" class="data row13 col17" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col18" class="data row13 col18" >-0.0012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col19" class="data row13 col19" >0.0017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col20" class="data row13 col20" >-0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col21" class="data row13 col21" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col22" class="data row13 col22" >0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col23" class="data row13 col23" >-0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col24" class="data row13 col24" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col25" class="data row13 col25" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col26" class="data row13 col26" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col27" class="data row13 col27" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col28" class="data row13 col28" >-0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col29" class="data row13 col29" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col30" class="data row13 col30" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col31" class="data row13 col31" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col32" class="data row13 col32" >0.33</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row13_col33" class="data row13 col33" >-0.093</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row14" class="row_heading level0 row14" >rate_government_action</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col0" class="data row14 col0" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col1" class="data row14 col1" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col2" class="data row14 col2" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col3" class="data row14 col3" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col4" class="data row14 col4" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col5" class="data row14 col5" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col6" class="data row14 col6" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col7" class="data row14 col7" >-0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col8" class="data row14 col8" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col9" class="data row14 col9" >-0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col10" class="data row14 col10" >-0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col11" class="data row14 col11" >-0.045</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col12" class="data row14 col12" >-0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col13" class="data row14 col13" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col14" class="data row14 col14" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col15" class="data row14 col15" >0.092</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col16" class="data row14 col16" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col17" class="data row14 col17" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col18" class="data row14 col18" >-0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col19" class="data row14 col19" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col20" class="data row14 col20" >-0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col21" class="data row14 col21" >-0.046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col22" class="data row14 col22" >-0.0042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col23" class="data row14 col23" >-0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col24" class="data row14 col24" >0.00059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col25" class="data row14 col25" >-0.0058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col26" class="data row14 col26" >-0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col27" class="data row14 col27" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col28" class="data row14 col28" >-0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col29" class="data row14 col29" >-0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col30" class="data row14 col30" >-0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col31" class="data row14 col31" >-0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col32" class="data row14 col32" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row14_col33" class="data row14 col33" >0.02</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row15" class="row_heading level0 row15" >rate_reducing_risk_single</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col0" class="data row15 col0" >0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col1" class="data row15 col1" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col2" class="data row15 col2" >-0.034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col3" class="data row15 col3" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col4" class="data row15 col4" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col5" class="data row15 col5" >-0.0085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col6" class="data row15 col6" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col7" class="data row15 col7" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col8" class="data row15 col8" >-0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col9" class="data row15 col9" >-0.063</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col10" class="data row15 col10" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col11" class="data row15 col11" >-0.054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col12" class="data row15 col12" >-0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col13" class="data row15 col13" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col14" class="data row15 col14" >0.092</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col15" class="data row15 col15" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col16" class="data row15 col16" >0.55</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col17" class="data row15 col17" >0.24</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col18" class="data row15 col18" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col19" class="data row15 col19" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col20" class="data row15 col20" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col21" class="data row15 col21" >0.005</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col22" class="data row15 col22" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col23" class="data row15 col23" >0.0016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col24" class="data row15 col24" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col25" class="data row15 col25" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col26" class="data row15 col26" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col27" class="data row15 col27" >-0.0079</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col28" class="data row15 col28" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col29" class="data row15 col29" >0.0093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col30" class="data row15 col30" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col31" class="data row15 col31" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col32" class="data row15 col32" >-0.45</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row15_col33" class="data row15 col33" >-0.038</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row16" class="row_heading level0 row16" >rate_reducing_risk_house</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col0" class="data row16 col0" >0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col1" class="data row16 col1" >0.0086</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col2" class="data row16 col2" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col3" class="data row16 col3" >0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col4" class="data row16 col4" >-0.0029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col5" class="data row16 col5" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col6" class="data row16 col6" >0.0059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col7" class="data row16 col7" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col8" class="data row16 col8" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col9" class="data row16 col9" >-0.049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col10" class="data row16 col10" >-0.042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col11" class="data row16 col11" >-0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col12" class="data row16 col12" >-0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col13" class="data row16 col13" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col14" class="data row16 col14" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col15" class="data row16 col15" >0.55</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col16" class="data row16 col16" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col17" class="data row16 col17" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col18" class="data row16 col18" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col19" class="data row16 col19" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col20" class="data row16 col20" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col21" class="data row16 col21" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col22" class="data row16 col22" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col23" class="data row16 col23" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col24" class="data row16 col24" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col25" class="data row16 col25" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col26" class="data row16 col26" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col27" class="data row16 col27" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col28" class="data row16 col28" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col29" class="data row16 col29" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col30" class="data row16 col30" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col31" class="data row16 col31" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col32" class="data row16 col32" >-0.37</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row16_col33" class="data row16 col33" >-0.008</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row17" class="row_heading level0 row17" >rate_reducing_mask</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col0" class="data row17 col0" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col1" class="data row17 col1" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col2" class="data row17 col2" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col3" class="data row17 col3" >-0.046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col4" class="data row17 col4" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col5" class="data row17 col5" >-0.0073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col6" class="data row17 col6" >-0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col7" class="data row17 col7" >-0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col8" class="data row17 col8" >-0.0091</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col9" class="data row17 col9" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col10" class="data row17 col10" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col11" class="data row17 col11" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col12" class="data row17 col12" >-0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col13" class="data row17 col13" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col14" class="data row17 col14" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col15" class="data row17 col15" >0.24</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col16" class="data row17 col16" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col17" class="data row17 col17" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col18" class="data row17 col18" >0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col19" class="data row17 col19" >0.00033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col20" class="data row17 col20" >0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col21" class="data row17 col21" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col22" class="data row17 col22" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col23" class="data row17 col23" >0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col24" class="data row17 col24" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col25" class="data row17 col25" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col26" class="data row17 col26" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col27" class="data row17 col27" >0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col28" class="data row17 col28" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col29" class="data row17 col29" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col30" class="data row17 col30" >0.069</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col31" class="data row17 col31" >0.12</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col32" class="data row17 col32" >-0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row17_col33" class="data row17 col33" >0.05</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row18" class="row_heading level0 row18" >covid19_positive</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col0" class="data row18 col0" >0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col1" class="data row18 col1" >0.0061</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col2" class="data row18 col2" >-0.0013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col3" class="data row18 col3" >0.007</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col4" class="data row18 col4" >0.0075</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col5" class="data row18 col5" >0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col6" class="data row18 col6" >-0.0041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col7" class="data row18 col7" >-0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col8" class="data row18 col8" >0.0024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col9" class="data row18 col9" >0.0068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col10" class="data row18 col10" >0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col11" class="data row18 col11" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col12" class="data row18 col12" >0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col13" class="data row18 col13" >-0.0012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col14" class="data row18 col14" >-0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col15" class="data row18 col15" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col16" class="data row18 col16" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col17" class="data row18 col17" >0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col18" class="data row18 col18" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col19" class="data row18 col19" >0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col20" class="data row18 col20" >0.051</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col21" class="data row18 col21" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col22" class="data row18 col22" >0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col23" class="data row18 col23" >0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col24" class="data row18 col24" >0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col25" class="data row18 col25" >0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col26" class="data row18 col26" >0.0057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col27" class="data row18 col27" >0.004</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col28" class="data row18 col28" >0.0041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col29" class="data row18 col29" >0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col30" class="data row18 col30" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col31" class="data row18 col31" >-0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col32" class="data row18 col32" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row18_col33" class="data row18 col33" >0.014</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row19" class="row_heading level0 row19" >covid19_symptoms</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col0" class="data row19 col0" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col1" class="data row19 col1" >0.0088</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col2" class="data row19 col2" >-0.003</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col3" class="data row19 col3" >-0.0065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col4" class="data row19 col4" >0.0036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col5" class="data row19 col5" >0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col6" class="data row19 col6" >-0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col7" class="data row19 col7" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col8" class="data row19 col8" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col9" class="data row19 col9" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col10" class="data row19 col10" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col11" class="data row19 col11" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col12" class="data row19 col12" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col13" class="data row19 col13" >0.0017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col14" class="data row19 col14" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col15" class="data row19 col15" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col16" class="data row19 col16" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col17" class="data row19 col17" >0.00033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col18" class="data row19 col18" >0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col19" class="data row19 col19" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col20" class="data row19 col20" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col21" class="data row19 col21" >0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col22" class="data row19 col22" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col23" class="data row19 col23" >0.042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col24" class="data row19 col24" >0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col25" class="data row19 col25" >0.009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col26" class="data row19 col26" >0.0023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col27" class="data row19 col27" >0.0012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col28" class="data row19 col28" >0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col29" class="data row19 col29" >0.043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col30" class="data row19 col30" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col31" class="data row19 col31" >0.00066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col32" class="data row19 col32" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row19_col33" class="data row19 col33" >-0.01</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row20" class="row_heading level0 row20" >covid19_contact</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col0" class="data row20 col0" >0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col1" class="data row20 col1" >-0.0084</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col2" class="data row20 col2" >0.007</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col3" class="data row20 col3" >-0.0055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col4" class="data row20 col4" >-0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col5" class="data row20 col5" >-0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col6" class="data row20 col6" >0.0058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col7" class="data row20 col7" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col8" class="data row20 col8" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col9" class="data row20 col9" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col10" class="data row20 col10" >0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col11" class="data row20 col11" >0.0088</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col12" class="data row20 col12" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col13" class="data row20 col13" >-0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col14" class="data row20 col14" >-0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col15" class="data row20 col15" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col16" class="data row20 col16" >-0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col17" class="data row20 col17" >0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col18" class="data row20 col18" >0.051</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col19" class="data row20 col19" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col20" class="data row20 col20" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col21" class="data row20 col21" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col22" class="data row20 col22" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col23" class="data row20 col23" >0.0057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col24" class="data row20 col24" >-0.0056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col25" class="data row20 col25" >-0.0038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col26" class="data row20 col26" >-0.0087</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col27" class="data row20 col27" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col28" class="data row20 col28" >-0.0079</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col29" class="data row20 col29" >0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col30" class="data row20 col30" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col31" class="data row20 col31" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col32" class="data row20 col32" >0.23</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row20_col33" class="data row20 col33" >-0.021</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row21" class="row_heading level0 row21" >asthma</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col0" class="data row21 col0" >0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col1" class="data row21 col1" >-0.001</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col2" class="data row21 col2" >-0.0089</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col3" class="data row21 col3" >-0.067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col4" class="data row21 col4" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col5" class="data row21 col5" >0.085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col6" class="data row21 col6" >-0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col7" class="data row21 col7" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col8" class="data row21 col8" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col9" class="data row21 col9" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col10" class="data row21 col10" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col11" class="data row21 col11" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col12" class="data row21 col12" >0.0011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col13" class="data row21 col13" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col14" class="data row21 col14" >-0.046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col15" class="data row21 col15" >0.005</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col16" class="data row21 col16" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col17" class="data row21 col17" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col18" class="data row21 col18" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col19" class="data row21 col19" >0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col20" class="data row21 col20" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col21" class="data row21 col21" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col22" class="data row21 col22" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col23" class="data row21 col23" >0.093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col24" class="data row21 col24" >0.009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col25" class="data row21 col25" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col26" class="data row21 col26" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col27" class="data row21 col27" >-0.00076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col28" class="data row21 col28" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col29" class="data row21 col29" >0.087</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col30" class="data row21 col30" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col31" class="data row21 col31" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col32" class="data row21 col32" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row21_col33" class="data row21 col33" >-0.014</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row22" class="row_heading level0 row22" >kidney_disease</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col0" class="data row22 col0" >-0.00053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col1" class="data row22 col1" >-0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col2" class="data row22 col2" >0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col3" class="data row22 col3" >0.0015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col4" class="data row22 col4" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col5" class="data row22 col5" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col6" class="data row22 col6" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col7" class="data row22 col7" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col8" class="data row22 col8" >0.0095</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col9" class="data row22 col9" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col10" class="data row22 col10" >0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col11" class="data row22 col11" >0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col12" class="data row22 col12" >-0.0036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col13" class="data row22 col13" >0.0027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col14" class="data row22 col14" >-0.0042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col15" class="data row22 col15" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col16" class="data row22 col16" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col17" class="data row22 col17" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col18" class="data row22 col18" >0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col19" class="data row22 col19" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col20" class="data row22 col20" >0.002</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col21" class="data row22 col21" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col22" class="data row22 col22" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col23" class="data row22 col23" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col24" class="data row22 col24" >0.081</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col25" class="data row22 col25" >0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col26" class="data row22 col26" >0.057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col27" class="data row22 col27" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col28" class="data row22 col28" >0.062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col29" class="data row22 col29" >0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col30" class="data row22 col30" >0.0056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col31" class="data row22 col31" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col32" class="data row22 col32" >0.00054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row22_col33" class="data row22 col33" >0.17</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row23" class="row_heading level0 row23" >compromised_immune</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col0" class="data row23 col0" >0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col1" class="data row23 col1" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col2" class="data row23 col2" >-0.003</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col3" class="data row23 col3" >-0.057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col4" class="data row23 col4" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col5" class="data row23 col5" >0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col6" class="data row23 col6" >-0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col7" class="data row23 col7" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col8" class="data row23 col8" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col9" class="data row23 col9" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col10" class="data row23 col10" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col11" class="data row23 col11" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col12" class="data row23 col12" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col13" class="data row23 col13" >-0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col14" class="data row23 col14" >-0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col15" class="data row23 col15" >0.0016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col16" class="data row23 col16" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col17" class="data row23 col17" >0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col18" class="data row23 col18" >0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col19" class="data row23 col19" >0.042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col20" class="data row23 col20" >0.0057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col21" class="data row23 col21" >0.093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col22" class="data row23 col22" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col23" class="data row23 col23" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col24" class="data row23 col24" >0.066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col25" class="data row23 col25" >0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col26" class="data row23 col26" >0.068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col27" class="data row23 col27" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col28" class="data row23 col28" >0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col29" class="data row23 col29" >0.26</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col30" class="data row23 col30" >0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col31" class="data row23 col31" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col32" class="data row23 col32" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row23_col33" class="data row23 col33" >0.093</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row24" class="row_heading level0 row24" >heart_disease</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col0" class="data row24 col0" >-0.0045</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col1" class="data row24 col1" >-5.9e-06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col2" class="data row24 col2" >0.00049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col3" class="data row24 col3" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col4" class="data row24 col4" >0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col5" class="data row24 col5" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col6" class="data row24 col6" >-0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col7" class="data row24 col7" >-0.0052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col8" class="data row24 col8" >0.0085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col9" class="data row24 col9" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col10" class="data row24 col10" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col11" class="data row24 col11" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col12" class="data row24 col12" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col13" class="data row24 col13" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col14" class="data row24 col14" >0.00059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col15" class="data row24 col15" >-0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col16" class="data row24 col16" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col17" class="data row24 col17" >0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col18" class="data row24 col18" >0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col19" class="data row24 col19" >0.0037</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col20" class="data row24 col20" >-0.0056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col21" class="data row24 col21" >0.009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col22" class="data row24 col22" >0.081</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col23" class="data row24 col23" >0.066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col24" class="data row24 col24" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col25" class="data row24 col25" >0.14</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col26" class="data row24 col26" >0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col27" class="data row24 col27" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col28" class="data row24 col28" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col29" class="data row24 col29" >0.06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col30" class="data row24 col30" >0.0026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col31" class="data row24 col31" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col32" class="data row24 col32" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row24_col33" class="data row24 col33" >0.38</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row25" class="row_heading level0 row25" >lung_disease</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col0" class="data row25 col0" >0.0018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col1" class="data row25 col1" >-0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col2" class="data row25 col2" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col3" class="data row25 col3" >-0.0065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col4" class="data row25 col4" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col5" class="data row25 col5" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col6" class="data row25 col6" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col7" class="data row25 col7" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col8" class="data row25 col8" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col9" class="data row25 col9" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col10" class="data row25 col10" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col11" class="data row25 col11" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col12" class="data row25 col12" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col13" class="data row25 col13" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col14" class="data row25 col14" >-0.0058</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col15" class="data row25 col15" >-0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col16" class="data row25 col16" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col17" class="data row25 col17" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col18" class="data row25 col18" >0.0033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col19" class="data row25 col19" >0.009</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col20" class="data row25 col20" >-0.0038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col21" class="data row25 col21" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col22" class="data row25 col22" >0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col23" class="data row25 col23" >0.09</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col24" class="data row25 col24" >0.14</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col25" class="data row25 col25" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col26" class="data row25 col26" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col27" class="data row25 col27" >0.0097</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col28" class="data row25 col28" >0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col29" class="data row25 col29" >0.073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col30" class="data row25 col30" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col31" class="data row25 col31" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col32" class="data row25 col32" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row25_col33" class="data row25 col33" >0.23</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row26" class="row_heading level0 row26" >diabetes</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col0" class="data row26 col0" >0.0016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col1" class="data row26 col1" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col2" class="data row26 col2" >-0.00014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col3" class="data row26 col3" >0.028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col4" class="data row26 col4" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col5" class="data row26 col5" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col6" class="data row26 col6" >-0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col7" class="data row26 col7" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col8" class="data row26 col8" >0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col9" class="data row26 col9" >0.0011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col10" class="data row26 col10" >0.0047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col11" class="data row26 col11" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col12" class="data row26 col12" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col13" class="data row26 col13" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col14" class="data row26 col14" >-0.0067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col15" class="data row26 col15" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col16" class="data row26 col16" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col17" class="data row26 col17" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col18" class="data row26 col18" >0.0057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col19" class="data row26 col19" >0.0023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col20" class="data row26 col20" >-0.0087</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col21" class="data row26 col21" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col22" class="data row26 col22" >0.057</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col23" class="data row26 col23" >0.068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col24" class="data row26 col24" >0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col25" class="data row26 col25" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col26" class="data row26 col26" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col27" class="data row26 col27" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col28" class="data row26 col28" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col29" class="data row26 col29" >0.068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col30" class="data row26 col30" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col31" class="data row26 col31" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col32" class="data row26 col32" >-0.0093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row26_col33" class="data row26 col33" >0.22</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row27" class="row_heading level0 row27" >hiv_positive</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col0" class="data row27 col0" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col1" class="data row27 col1" >0.00038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col2" class="data row27 col2" >-0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col3" class="data row27 col3" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col4" class="data row27 col4" >0.0094</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col5" class="data row27 col5" >-0.0066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col6" class="data row27 col6" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col7" class="data row27 col7" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col8" class="data row27 col8" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col9" class="data row27 col9" >0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col10" class="data row27 col10" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col11" class="data row27 col11" >0.064</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col12" class="data row27 col12" >-0.0055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col13" class="data row27 col13" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col14" class="data row27 col14" >-0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col15" class="data row27 col15" >-0.0079</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col16" class="data row27 col16" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col17" class="data row27 col17" >0.0044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col18" class="data row27 col18" >0.004</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col19" class="data row27 col19" >0.0012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col20" class="data row27 col20" >0.0031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col21" class="data row27 col21" >-0.00076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col22" class="data row27 col22" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col23" class="data row27 col23" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col24" class="data row27 col24" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col25" class="data row27 col25" >0.0097</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col26" class="data row27 col26" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col27" class="data row27 col27" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col28" class="data row27 col28" >0.0074</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col29" class="data row27 col29" >-0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col30" class="data row27 col30" >0.0066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col31" class="data row27 col31" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col32" class="data row27 col32" >-0.0021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row27_col33" class="data row27 col33" >0.03</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row28" class="row_heading level0 row28" >hypertension</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col0" class="data row28 col0" >-0.0054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col1" class="data row28 col1" >-0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col2" class="data row28 col2" >0.0021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col3" class="data row28 col3" >0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col4" class="data row28 col4" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col5" class="data row28 col5" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col6" class="data row28 col6" >-0.0078</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col7" class="data row28 col7" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col8" class="data row28 col8" >0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col9" class="data row28 col9" >-0.0064</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col10" class="data row28 col10" >-0.0038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col11" class="data row28 col11" >-0.017</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col12" class="data row28 col12" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col13" class="data row28 col13" >-0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col14" class="data row28 col14" >-0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col15" class="data row28 col15" >-0.011</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col16" class="data row28 col16" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col17" class="data row28 col17" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col18" class="data row28 col18" >0.0041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col19" class="data row28 col19" >0.0019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col20" class="data row28 col20" >-0.0079</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col21" class="data row28 col21" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col22" class="data row28 col22" >0.062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col23" class="data row28 col23" >0.056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col24" class="data row28 col24" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col25" class="data row28 col25" >0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col26" class="data row28 col26" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col27" class="data row28 col27" >0.0074</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col28" class="data row28 col28" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col29" class="data row28 col29" >0.067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col30" class="data row28 col30" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col31" class="data row28 col31" >0.14</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col32" class="data row28 col32" >-0.0094</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row28_col33" class="data row28 col33" >0.28</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row29" class="row_heading level0 row29" >other_chronic</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col0" class="data row29 col0" >0.0042</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col1" class="data row29 col1" >0.00099</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col2" class="data row29 col2" >-0.0082</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col3" class="data row29 col3" >-0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col4" class="data row29 col4" >0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col5" class="data row29 col5" >0.049</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col6" class="data row29 col6" >-0.053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col7" class="data row29 col7" >0.03</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col8" class="data row29 col8" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col9" class="data row29 col9" >0.0014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col10" class="data row29 col10" >0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col11" class="data row29 col11" >0.0028</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col12" class="data row29 col12" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col13" class="data row29 col13" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col14" class="data row29 col14" >-0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col15" class="data row29 col15" >0.0093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col16" class="data row29 col16" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col17" class="data row29 col17" >0.047</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col18" class="data row29 col18" >0.0046</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col19" class="data row29 col19" >0.043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col20" class="data row29 col20" >0.0071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col21" class="data row29 col21" >0.087</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col22" class="data row29 col22" >0.044</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col23" class="data row29 col23" >0.26</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col24" class="data row29 col24" >0.06</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col25" class="data row29 col25" >0.073</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col26" class="data row29 col26" >0.068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col27" class="data row29 col27" >-0.0034</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col28" class="data row29 col28" >0.067</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col29" class="data row29 col29" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col30" class="data row29 col30" >0.062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col31" class="data row29 col31" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col32" class="data row29 col32" >-0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row29_col33" class="data row29 col33" >0.093</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row30" class="row_heading level0 row30" >opinion_infection</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col0" class="data row30 col0" >0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col1" class="data row30 col1" >-0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col2" class="data row30 col2" >0.00022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col3" class="data row30 col3" >-0.031</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col4" class="data row30 col4" >0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col5" class="data row30 col5" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col6" class="data row30 col6" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col7" class="data row30 col7" >0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col8" class="data row30 col8" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col9" class="data row30 col9" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col10" class="data row30 col10" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col11" class="data row30 col11" >0.023</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col12" class="data row30 col12" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col13" class="data row30 col13" >0.013</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col14" class="data row30 col14" >-0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col15" class="data row30 col15" >0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col16" class="data row30 col16" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col17" class="data row30 col17" >0.069</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col18" class="data row30 col18" >0.055</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col19" class="data row30 col19" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col20" class="data row30 col20" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col21" class="data row30 col21" >0.071</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col22" class="data row30 col22" >0.0056</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col23" class="data row30 col23" >0.059</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col24" class="data row30 col24" >0.0026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col25" class="data row30 col25" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col26" class="data row30 col26" >0.016</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col27" class="data row30 col27" >0.0066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col28" class="data row30 col28" >0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col29" class="data row30 col29" >0.062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col30" class="data row30 col30" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col31" class="data row30 col31" >0.24</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col32" class="data row30 col32" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row30_col33" class="data row30 col33" >-0.065</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row31" class="row_heading level0 row31" >opinion_mortality</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col0" class="data row31 col0" >0.0072</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col1" class="data row31 col1" >-0.0062</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col2" class="data row31 col2" >-0.00086</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col3" class="data row31 col3" >-0.096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col4" class="data row31 col4" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col5" class="data row31 col5" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col6" class="data row31 col6" >-0.076</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col7" class="data row31 col7" >0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col8" class="data row31 col8" >0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col9" class="data row31 col9" >0.0096</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col10" class="data row31 col10" >0.0097</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col11" class="data row31 col11" >-0.0032</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col12" class="data row31 col12" >-0.039</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col13" class="data row31 col13" >-0.018</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col14" class="data row31 col14" >-0.083</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col15" class="data row31 col15" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col16" class="data row31 col16" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col17" class="data row31 col17" >0.12</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col18" class="data row31 col18" >-0.0025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col19" class="data row31 col19" >0.00066</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col20" class="data row31 col20" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col21" class="data row31 col21" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col22" class="data row31 col22" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col23" class="data row31 col23" >0.21</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col24" class="data row31 col24" >0.11</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col25" class="data row31 col25" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col26" class="data row31 col26" >0.16</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col27" class="data row31 col27" >0.026</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col28" class="data row31 col28" >0.14</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col29" class="data row31 col29" >0.18</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col30" class="data row31 col30" >0.24</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col31" class="data row31 col31" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col32" class="data row31 col32" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row31_col33" class="data row31 col33" >0.15</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row32" class="row_heading level0 row32" >risk_infection</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col0" class="data row32 col0" >0.025</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col1" class="data row32 col1" >-0.04</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col2" class="data row32 col2" >0.1</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col3" class="data row32 col3" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col4" class="data row32 col4" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col5" class="data row32 col5" >0.052</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col6" class="data row32 col6" >-0.022</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col7" class="data row32 col7" >-0.0053</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col8" class="data row32 col8" >0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col9" class="data row32 col9" >0.041</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col10" class="data row32 col10" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col11" class="data row32 col11" >0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col12" class="data row32 col12" >0.62</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col13" class="data row32 col13" >0.33</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col14" class="data row32 col14" >-0.035</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col15" class="data row32 col15" >-0.45</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col16" class="data row32 col16" >-0.37</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col17" class="data row32 col17" >-0.13</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col18" class="data row32 col18" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col19" class="data row32 col19" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col20" class="data row32 col20" >0.23</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col21" class="data row32 col21" >-0.0043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col22" class="data row32 col22" >0.00054</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col23" class="data row32 col23" >-0.029</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col24" class="data row32 col24" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col25" class="data row32 col25" >-0.019</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col26" class="data row32 col26" >-0.0093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col27" class="data row32 col27" >-0.0021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col28" class="data row32 col28" >-0.0094</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col29" class="data row32 col29" >-0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col30" class="data row32 col30" >0.2</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col31" class="data row32 col31" >-0.048</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col32" class="data row32 col32" >1.0</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row32_col33" class="data row32 col33" >-0.068</td>
            </tr>
            <tr>
                        <th id="T_fced5690_9082_11ea_88ae_54e1adf13a74level0_row33" class="row_heading level0 row33" >risk_mortality</th>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col0" class="data row33 col0" >-0.004</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col1" class="data row33 col1" >-0.012</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col2" class="data row33 col2" >-0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col3" class="data row33 col3" >0.085</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col4" class="data row33 col4" >0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col5" class="data row33 col5" >0.027</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col6" class="data row33 col6" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col7" class="data row33 col7" >-0.043</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col8" class="data row33 col8" >0.015</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col9" class="data row33 col9" >0.033</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col10" class="data row33 col10" >0.036</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col11" class="data row33 col11" >0.024</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col12" class="data row33 col12" >-0.089</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col13" class="data row33 col13" >-0.093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col14" class="data row33 col14" >0.02</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col15" class="data row33 col15" >-0.038</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col16" class="data row33 col16" >-0.008</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col17" class="data row33 col17" >0.05</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col18" class="data row33 col18" >0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col19" class="data row33 col19" >-0.01</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col20" class="data row33 col20" >-0.021</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col21" class="data row33 col21" >-0.014</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col22" class="data row33 col22" >0.17</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col23" class="data row33 col23" >0.093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col24" class="data row33 col24" >0.38</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col25" class="data row33 col25" >0.23</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col26" class="data row33 col26" >0.22</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col27" class="data row33 col27" >0.03</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col28" class="data row33 col28" >0.28</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col29" class="data row33 col29" >0.093</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col30" class="data row33 col30" >-0.065</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col31" class="data row33 col31" >0.15</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col32" class="data row33 col32" >-0.068</td>
                        <td id="T_fced5690_9082_11ea_88ae_54e1adf13a74row33_col33" class="data row33 col33" >1.0</td>
            </tr>
    </tbody></table>



### Raw Data Inspection Observations: 

> Most of the data collected ~ 87% comes from the United states with Canada 5% and UK 2.5% next.  The rest of the countries reporting are even smaller in terms of contribution size.  A very small percentage: **.0014% tested positive for COVID-19** in this sample.  There are no direct correlations and the most highly correlated features of the unprocessed data are: 

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


Initial visual inspection of null values indicates that region and prescription medication are sparsely populated.  Since region was ~90% missing, it was dropped.  Prescription medication had 57K values and details are [included in this section](#Prescription-Medication). 

The opinion_infections and opinion_mortality columns are also a little 'light' in terms of responses and have the same number of responses.  This null rate of ~16% was imputed with the median values for each respective field. 

Null values in columns that contain <5% null values were dropped.  

Other than those outlined above, there doesn't seem to be be any other apparent patterns for incomplete data. (See below).


### Null or Missing Data: 


```python
import missingno
missingno.matrix(df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x22780ff8b38>




![png](output_25_1.png)


Aditional inspection shows that there are quite a few columns with less than 5% null values.  Since this dataset is so large, it seems reasonable to remove these.  Details follow:


```python
nulls = pd.DataFrame(df.isna().sum()/len(df)*100)
nulls = pd.DataFrame(nulls.reset_index())
nulls.columns=['variable', '%_Null']
nulls.sort_values(by='%_Null', ascending=False, inplace=True)
nulls

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>%_Null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>region</td>
      <td>93.167342</td>
    </tr>
    <tr>
      <th>38</th>
      <td>prescription_medication</td>
      <td>68.800876</td>
    </tr>
    <tr>
      <th>40</th>
      <td>opinion_mortality</td>
      <td>17.445604</td>
    </tr>
    <tr>
      <th>39</th>
      <td>opinion_infection</td>
      <td>17.445604</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cocaine</td>
      <td>4.705611</td>
    </tr>
    <tr>
      <th>15</th>
      <td>amphetamines</td>
      <td>4.430825</td>
    </tr>
    <tr>
      <th>17</th>
      <td>lsd</td>
      <td>4.089644</td>
    </tr>
    <tr>
      <th>18</th>
      <td>mdma</td>
      <td>3.513255</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cannabis</td>
      <td>2.017198</td>
    </tr>
    <tr>
      <th>21</th>
      <td>text_working</td>
      <td>0.683654</td>
    </tr>
    <tr>
      <th>19</th>
      <td>contacts_count</td>
      <td>0.683654</td>
    </tr>
    <tr>
      <th>25</th>
      <td>rate_reducing_mask</td>
      <td>0.299341</td>
    </tr>
    <tr>
      <th>12</th>
      <td>smoking</td>
      <td>0.299341</td>
    </tr>
    <tr>
      <th>13</th>
      <td>alcohol</td>
      <td>0.299341</td>
    </tr>
    <tr>
      <th>41</th>
      <td>risk_infection</td>
      <td>0.012439</td>
    </tr>
    <tr>
      <th>42</th>
      <td>risk_mortality</td>
      <td>0.012439</td>
    </tr>
    <tr>
      <th>2</th>
      <td>country</td>
      <td>0.002746</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ip_accuracy</td>
      <td>0.000162</td>
    </tr>
    <tr>
      <th>11</th>
      <td>blood_type</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>kidney_disease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ip_latitude</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ip_longitude</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>other_chronic</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>hypertension</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>hiv_positive</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>diabetes</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>lung_disease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>heart_disease</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>compromised_immune</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>asthma</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bmi</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>covid19_contact</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>covid19_symptoms</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>covid19_positive</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sex</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>rate_reducing_risk_house</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>rate_reducing_risk_single</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>rate_government_action</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>age</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>house_count</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>height</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>weight</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>survey_date</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



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




![png](output_38_2.png)


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
2. Practice social distancing and hygiene individually and as a household
        *This is especially true for males and those with high BMI.
3. Avoid contact with those known to be infected.
4.  If taking prescription medication, it is recommended to discuss additional risk for infection with a physician.

    
 



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
