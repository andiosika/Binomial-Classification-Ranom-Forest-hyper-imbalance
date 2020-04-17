


###plotting the importance of features based
### on 
def plot_importance(clf, top_n=20,figsize=(15,12)):
    df_importance = pd.Series(tree.feature_importances_,index=X_train.columns)
    df_importance.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',figsize=figsize)
    return df_importance
plot_importance(tree);

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
    import seaborn as sns
    """Returns vizual information on
    outliers, value counts and description
    
    args:
    col - df['column']
    """
    fig, ax = plt.subplots(figsize=(6,4),ncols=1)

    x = sns.boxplot(col)
    y = col.value_counts()
    z = col.describe()
    
    return x,y,z