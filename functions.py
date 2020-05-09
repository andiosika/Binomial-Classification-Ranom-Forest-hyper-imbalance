


def des(feature):
    ''' args: 
        col - df['column'], the column name from 
    a pd.DataFrame

    Takes a feature from a dataframe and provides value counts, 
    high-level information and provides a visual to identify 
    outliers'''
    import seaborn as sns
    
    y = feature.value_counts()
    z = feature.describe()

    display( "------------- Value Counts: -----------"
            ,y, "-------------- Description: ---------"
            , z)
    
#     x = sns.boxplot(feature)
    
#     return x


###plotting the importance of features based
### on decision tree model visually and with a 
### DataFrame
def plot_importance(clf, X_train, top_n=20,figsize=(15,12)):
    import pandas as pd
    """args:
    classification model - designed on DecisionTreeClassification

    top_n - (default=20) The count of features requested to display
    
    figsize - (default(15,20) The figure size required to best visually display 
    this information.

    Plots the top features affecting categorization
    using .feature_importance_ from the corresponding classification 
    model"""

    df_import = pd.Series(clf.feature_importances_, index=X_train.columns, name='importance')
    df_import.sort_values().tail(top_n).plot(kind='barh', figsize=(10,10))
    return df_import.sort_values(ascending=False)

def df_import(clf, X_train, n=20):  
     """args:
    classification model - designed on DecisionTreeClassification

    Training data

    n - (default=20) The count of features requested to display
    
    figsize - (default(15,20) The figure size required to best visually display 
    this information.

    Visually displays the top features affecting categorization
    using .feature_importance_ from the corresponding classification 
    model
    """
    import pandas as pd

    df_import = pd.DataFrame(clf.feature_importances_, X_train.columns)
    df_import.reset_index(inplace=True)
    df_import.columns = ['factor', 'importance']
    x = df_import.sort_values('importance', ascending=False).head(n)
    x.reset_index(drop=True,inplace=True)
    y = x.sort_values('importance', ascending=False).head(n).style.bar(subset=['importance'], color='#d65f5f')
    
    return y
    



## function to evalute the model
def evaluate_model(X_true, y_true, y_pred, X_train, y_train, clf):
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    import pandas as pd
    ''' Evaluation tool to evaluate accuracy of a given model, 
    
    y_true - typically the y_test, untrained values of a dataset obtained in
    a train/test split
    
    y_pred - the y_values predicted by a given model
    X_true - typically the X_test values obtained in standard train/test
    split
    
    returns a classification report detailing the f1, and accuracy score
    along with visuals to inform results including a confusion matrix and
    ROC Curve'''

       ## Classification Report / Scores 
    
    report = metrics.classification_report(y_true,y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    
    display(report_df)

    ###Testing Accuracy Training Accuracy
    print("___________________________________________")
    print("\n")
    print("Training Accuracy : ", clf.score(X_train, y_train))
    print("Test Accuracy : ", clf.score(X_true, y_true))
    print("___________________________________________")
    
  
#     import seaborn as sns
#     sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    
    fig, ax = plt.subplots(figsize=(10,4),ncols=2)

    # ax[0].axis('off')
    # ax[0].axis('tight')
    # ax[0].table(cellText=report_df.values,
    #          colLabels=report_df.columns,
    #          rowLabels=report_df.index,
    #          loc='center',
    #          bbox=[0.2, 0.2, 0.8, 0.8])


    metrics.plot_confusion_matrix(clf,X_true,y_true,cmap='gnuplot',
                                  normalize='true',ax=ax[0])
    ax[0].set(title='Confusion Matrix')
    y_score = clf.predict_proba(X_true)[:,1]

    fpr,tpr,thresh = metrics.roc_curve(y_true,y_score)
    
    #print(f'AUC: {(fpr, tpr, thresh)}')
    roc_auc = round(metrics.auc(fpr,tpr),3)
    ax[1].plot(fpr,tpr,color='r',label=f'AUC={roc_auc}')
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].legend()
    ax[1].grid()
    ax[1].set(ylabel='True Positive Rate',xlabel='False Positive Rate',
          title='Receiver operating characteristic (ROC) Curve')
    plt.tight_layout()
    plt.show()


    

#quick vizual on columns
def bxplt(col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    """Returns vizual information on
    outliers, value counts and description
    
    args:
    col - df['column'], the column name from 
    a pd.DataFrame
    """
    import seaborn as sns
    
    y = col.value_counts()
    z = col.describe()

    display( "------------- Value Counts: -----------",y, "-------------- Description: ---------" , z)
    
    x = sns.boxplot(col)
    
    return x
    

    ## a timer to record how long a process takes
class Timer():
'''A timer used to record how long a process takes.

After instaniating, a .start() and .stop() can be used 
before and after a process in respective order.'''




    ## def init
    def __init__(self,format_="%m/%d/%y - %I:%M %p"):
        import tzlocal
        self.tz = tzlocal.get_localzone()
        self.fmt = format_
        
        self.created_at = self.get_time()# get time
        

    
    ## def get time method
    def get_time(self):
        import datetime as dt
        return dt.datetime.now(self.tz)

    ## def start
    def start(self):
        time = self.get_time()
        self.start = time
        print(f"[i] Timer started at{self.start.strftime(self.fmt)}")
        
        ## def stop
    def stop(self):
        time = self.get_time()
        self.end = time
        print(f"[i] Timer ended at {self.end.strftime(self.fmt)}")
        print(f"- Total time = {self.end-self.start}")
timer = Timer()
print(timer.created_at)
timer.start()
timer.stop()



def bxplt(col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    """Returns vizual information on
    outliers, value counts and description
    
    args:
    col - df['column'], the column name from 
    a pd.DataFrame
    """
    import seaborn as sns
    
    y = col.value_counts()
    z = col.describe()
    
    x = sns.boxplot(col)

    return y, z , x