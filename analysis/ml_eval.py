import numpy as np
import sklearn.metrics as skmet
import pandas as pd


def get_classification_metrics(model, X_test, y_test):
        """A function to 'pprint' classification metrics (binary)
        
        Args:
            - model (trainned model with .predict_proba() method): a learned model
            - y_test (pd.DataFame): column name used to evaluate model performance
            - X_test (pd.DataFame of str): list of features - must match training
            
        Returns:
            - res (pd.dataframe): results table
        """

        y_proba = model.predict_proba(X_test)[:,1]

        # compute tpr/fpr at every thresh
        fpr, tpr, thresholds = skmet.roc_curve(y_test, 
                                               y_proba, 
                                               pos_label=model.classes_[1])

        # get optimal threshold by AUCROC - by Youden's J
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        aucroc = skmet.roc_auc_score(y_test, y_proba)

        # compute predictions based on optimal threshold
        y_pred = np.where(y_proba >= optimal_threshold, 1, 0)
        cm = skmet.confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # get precision/recall
        rate = y_test.mean()
        precision = tp / (tp + fn * (1 / rate - 1))
        recall = tp / (tp + fn * (1 / rate - 1))
        f1 = 2 * tp / (2*tp + fp + fn)

        res_dict = {
            'optimal_threshold':optimal_threshold,
            'true negatives': tn,
            'true positives': tp,
            'false positives': fp,
            'false negatives': fn,
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'F1-score' : f1,
            'precision': precision,
            'recall': recall,
            'AUCROC' : aucroc,
        }

        res = pd.DataFrame.from_dict(res_dict, orient='index')
        return res