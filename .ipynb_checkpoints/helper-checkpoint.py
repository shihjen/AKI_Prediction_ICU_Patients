# helper function to perform hyperparameters tuning via grid search and return the tuned model
def hyperparameterTuning(model, param_grid, selected_feats, Xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    stratifiedCV = StratifiedKFold(n_splits=5)
    # create the GridSearchCV object
    grid_search = GridSearchCV(model,                  # model to be tuned
                               param_grid,             # search grid for the parameters
                               cv=stratifiedCV,        # stratified K-fold cross validation to evaluate the model performance
                               scoring='roc_auc',      # metric to assess the model performance, weighted F1 score (consider the proportion of classes in the dataset)
                               n_jobs=-1)              # use all cpu cores to speed-up CV search

    # fit the data into the grid search space
    grid_search.fit(Xtrain[selected_feats], ytrain)

    # print the best parameters and the corresponding ROC_AUC score
    print('Best Hyperparameters from Grid Search : ', grid_search.best_params_)
    print('Best Weighted F1 Score: ', grid_search.best_score_)
    print()

    # get the best model
    best_model = grid_search.best_estimator_
    
    # return the hyperparameters tuned model
    return best_model



# helper function to check model performance
# function to display classification report, ROC curve, and confusion matrix
def modelPerformance(model, Xtrain, Xtest, ytrain, ytest):
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    # to predict classes
    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)
    
    # to predict probabilities
    ypred_train_proba = model.predict_proba(Xtrain)
    ypred_test_proba = model.predict_proba(Xtest)

    labels = ['No AKI','AKI']

    # classification report (for metrics: precision, recall, F1, & accuracy)
    report_train = classification_report(ytrain, ypred_train, target_names=labels)
    report_test = classification_report(ytest, ypred_test, target_names=labels)

    # print the classification report 
    print('Classification report for training data:')
    print(report_train)
    print('Classification report for test data:')
    print(report_test)

    # AUC-ROC curve
    # plot the ROC curve
    fpr_train, tpr_train, _ = roc_curve(ytrain, ypred_train_proba[:,1])
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(ytest, ypred_test_proba[:,1])
    roc_auc_test = auc(fpr_test, tpr_test)
    
    plt.figure(figsize=(12, 6))
    plt.plot(fpr_train, tpr_train, color='indigo', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='lightcoral', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20)
    plt.legend(loc='lower right')
    plt.show()
    
    # precision-recall curve
    precision_train, recall_train, thresholds_train = precision_recall_curve(ytrain, ypred_train_proba[:,1])
    precision_test, recall_test, thresholds_test = precision_recall_curve(ytest, ypred_test_proba[:,1])
    
    # compute average precision for train and test sets
    ap_train = average_precision_score(ytrain, ypred_train_proba[:, 1])
    ap_test = average_precision_score(ytest, ypred_test_proba[:, 1])
    
    plt.figure(figsize=(12, 6))
    plt.plot(recall_train, precision_train, color='indigo', lw=2, label=f'Train Precision-Recall Curve (AP = {ap_train:.2f})')
    plt.plot(recall_test, precision_test, color='lightcoral', lw=2, label=f'Test Precision-Recall Curve (AP = {ap_test:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=20)
    plt.legend(loc='lower right')
    plt.show()

    # diagnosis - confusion matrix
    cm_train = confusion_matrix(ytrain, ypred_train, normalize='true')
    cm_test = confusion_matrix(ytest, ypred_test, normalize='true')

    # visualize confusion matrix in heatmap
    figure, axes = plt.subplots(1,2, figsize=(13,5))
    sns.heatmap(cm_train, annot=True, cmap='BuPu', xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[0])
    axes[0].set_title('Training', fontsize=15)
    sns.heatmap(cm_test, annot=True, cmap='BuPu', xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[1])
    axes[1].set_title('Test', fontsize=15)
    figure.suptitle('Confusion Matrix', fontsize=20)
    plt.tight_layout(pad=2)
    plt.show()


def eval_metrics(model, Xtest, ytest, description):
    import pandas as pd
    from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix
    ypred_test = model.predict(Xtest)
    ypred_test_proba = model.predict_proba(Xtest)

    # AUROC score
    fpr, tpr, threshold = roc_curve(ytest, ypred_test_proba[:,1])
    roc_auc = auc(fpr, tpr)

    # average precision score
    average_precision = average_precision_score(ytest, ypred_test_proba[:,1])

    # recall score for AKI
    recall = recall_score(ytest, ypred_test, average='macro')

    # precision score for AKI
    precision = precision_score(ytest, ypred_test, average='macro')
    
    # F1 score for AKI
    f1 = f1_score(ytest, ypred_test, average='macro')

    # false positive rate
    tn, fp, fn, tp = confusion_matrix(ytest, ypred_test, labels=[0, 1]).ravel()
    false_positive_rate = (fp / (fp + tn))

    result = pd.DataFrame({'AUROC':roc_auc,
                                       'Average_Precision':average_precision,
                                       'Precision':precision,
                                       'Recall':recall,
                                       'F1 score':f1,
                                       'False Positive Rate':false_positive_rate}, 
                                      index=[description])
    
    return result


# helper function to perform recursive feature elimination
# function take in 4 arguments, X=feature matrix, y=target, RFE_estimator=learning model to provide feature importance (weights, feature importance)
# num_feats=number of features return (default 10 features)
def recursiveFeatureSelection(X,y,RFE_estimator,num_feats=10,verbose=True):
    from sklearn.feature_selection import RFE
    selector = RFE(estimator=RFE_estimator, n_features_to_select=num_feats)
    selector.fit(X,y)
    selected_feats = selector.get_feature_names_out()
    if verbose:
        print('Features selected via recursive feature elimination approach:',', '.join(selected_feats))
    return selected_feats


# helper function to perform stepwise feature selection (forward or backward)
# function take in 5 arguments: X=feature matrix, y=target, C=inverse regularization strength in logistic regression (default=0.1)
# num_feats=number of features return (default 10 features), direction=forward or backward selection (default=forward)
def sequentialFeatureSelection(X,y,sfs_estimator,num_feats=10,direction='forward',verbose=True):
    from sklearn.feature_selection import SequentialFeatureSelector
    selector = SequentialFeatureSelector(estimator=sfs_estimator, n_features_to_select=num_feats, cv=5, direction=direction, n_jobs=-1)
    selector.fit(X,y)
    selected_feats = selector.get_feature_names_out()
    if verbose:
        print('%d features selected via stepwise forward selection approach:'%num_feats,', '.join(selected_feats))
        print()
    return selected_feats



# helper function to perform feature selection via genetic algorithm
def gaFeatureSelection(estimator,X,y,population_size=10,generations=10,verbose=False):
    from sklearn_genetic import GAFeatureSelectionCV
    evolved_estimator = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring='roc_auc',
        population_size=population_size,
        generations=generations,
        n_jobs=-1,
        verbose=True,
        keep_top_k=2,
        elitism=True,
    )

    evolved_estimator.fit(X,y)
    support = evolved_estimator.support_
    selected_feats = list(X.columns[support])
    if verbose:
        print('\n{} features are selected via genetic algorithm.'.format(len(selected_feats)))
        print('These features are:',', '.join(selected_feats),'\n')
    return selected_feats