


###plotting the importance of features based
### on decision tree model visually and with a 
### DataFrame
def plot_importance(clf, top_n=20,figsize=(15,12)):
    """args:
    classification model - designed on DecisionTreeClassification

    top_n - (default=20) The count of features requested to display
    
    figsize - (default(15,20) The figure size required to best visually display 
    this information.

    Plots the top features affecting categorization
    using .feature_importance_ from the corresponding classification 
    model
    """
    df_importance = pd.Series(clf.feature_importances_,index=X_train.columns)
    df_importance.sort_values(ascending=False).head(top_n).plot(
        kind='barh',figsize=figsize)
    return df_importance

def df_importance(tree, top_n=20):
    df_import = pd.DataFrame(tree.feature_importances_, X_train.columns)
    df_import.reset_index(inplace=True)
    df_import.columns = ['feature', 'coef']
    df_import.sort_values('coef', ascending=False).head(top_n)


## function to evalute the model
def evaluate_model(y_true, y_pred,X_true,clf):
    import sklearn.metrics as metrics
    """ Evaluation tool to evaluate accuracy of a given model, 
    
    Args: 
    y_true - typically the y_test, untrained values of a dataset obtained in
    a train/test split
    
    y_pred - the y_values predicted by a given model
    X_true - typically the X_test values obtained in standard train/test
    split
    
    returns:
     Classification report detailing the f1 and accuracy score
    along with visuals to inform results"""
    
    ## Classification Report / Scores 
    print(metrics.classification_report(y_true,y_pred))

    fig, ax = plt.subplots(figsize=(12,6),ncols=2)
    metrics.plot_confusion_matrix(clf,X_true,y_true,cmap="Blues",
                                  normalize='true',ax=ax[0])
    ax[0].set(title='Confusion Matrix')
    y_score = clf.predict_proba(X_true)[:,1]

    fpr,tpr,thresh = metrics.roc_curve(y_true,y_score)
    # print(f"ROC-area-under-the-curve= {}")
    roc_auc = round(metrics.auc(fpr,tpr),3)
    ax[1].plot(fpr,tpr,color='darkorange',label=f'AUC={roc_auc}')
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].legend()
    ax[1].grid()
    ax[1].set(ylabel='True Positive Rate',xlabel='False Positive Rate',
          title='Receiver operating characteristic (ROC) Curve')
    plt.tight_layout()
    plt.show()
    try: 
        df_important = plot_importance(clf)
    except:
        df_important = None
    

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
    
    y = feature.value_counts()
    z = feature.describe()

    display( "------------- Value Counts: -----------",y, "-------------- Description: ---------" , z)
    
    x = sns.boxplot(col)
    
    return x
    

    ## a timer to record how long a process takes
# class Timer():
#     ## def init
#     def __init__(self,format_="%m/%d/%y - %I:%M %p"):
#         import tzlocal
#         self.tz = tzlocal.get_localzone()
#         self.fmt = format_
        
#         self.created_at = self.get_time()# get time
        

    
#     ## def get time method
#     def get_time(self):
#         import datetime as dt
#         return dt.datetime.now(self.tz)

#     ## def start
#     def start(self):
#         time = self.get_time()
#         self.start = time
#         print(f"[i] Timer started at{self.start.strftime(self.fmt)}")
        
#         ## def stop
#     def stop(self):
#         time = self.get_time()
#         self.end = time
#         print(f"[i] Timer ended at {self.end.strftime(self.fmt)}")
#         print(f"- Total time = {self.end-self.start}")
# timer = Timer()
# print(timer.created_at)
# timer.start()
# timer.stop()