


def des(feature):
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
    model
    """
    df_import = pd.Series(clf.feature_importances_, index=X_train.columns, name='importance').head(top_n)
    df_import.sort_values().plot(kind='barh', figsize=(15,12))
    return df_importance

def df_importance(tree, X_train, top_n=20):
    import pandas as pd

    df_import = pd.DataFrame(tree.feature_importances_, X_train.columns)
    df_import.reset_index(inplace=True)
    df_import.columns = ['feature', 'coef']
    df_import.sort_values('coef', ascending=False).head(top_n)



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
    along with visuals to inform results'''

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