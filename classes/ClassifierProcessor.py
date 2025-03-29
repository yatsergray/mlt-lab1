import numpy as np
from pandas import DataFrame, Series
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, precision_recall_curve, roc_curve, \
    accuracy_score, auc


class ClassifierProcessor:

    @staticmethod
    def fit_classifiers(classifier_models: list, x_train: DataFrame, y_train: Series):
        for classifier_model in classifier_models:
            classifier_model.classifier.fit(x_train, y_train)

    @staticmethod
    def evaluate_classifier(classifier_model, x_test: DataFrame, y_test: Series, use_threshold: bool = False):
        if use_threshold:
            y_pred = (classifier_model.classifier.predict_proba(x_test)[:,
                      1] > classifier_model.optimal_threshold).astype(int)
        else:
            y_pred = classifier_model.classifier.predict(x_test)

        classifier_model.accuracy = accuracy_score(y_test, y_pred)
        classifier_model.precision = precision_score(y_test, y_pred)
        classifier_model.recall = recall_score(y_test, y_pred)
        classifier_model.f1 = f1_score(y_test, y_pred)
        classifier_model.loss = log_loss(y_test, y_pred)

        return classifier_model

    @staticmethod
    def calculate_precisions_recalls_and_roc_curves_and_optimal_threshold(classifier_model, x_test: DataFrame,
                                                                          y_test: Series):
        y_prob = classifier_model.classifier.predict_proba(x_test)[:, 1]

        classifier_model_precisions, classifier_model_recalls, _ = precision_recall_curve(y_test, y_prob)
        classifier_model_fp_rates, classifier_model_tp_rates, thresholds = roc_curve(y_test, y_prob)

        classifier_model_optimal_threshold = thresholds[
            np.argmax(classifier_model_tp_rates - classifier_model_fp_rates)]
        classifier_model_auc = auc(classifier_model_fp_rates, classifier_model_tp_rates)

        classifier_model.precisions = classifier_model_precisions
        classifier_model.recalls = classifier_model_recalls
        classifier_model.fp_rates = classifier_model_fp_rates
        classifier_model.tp_rates = classifier_model_tp_rates
        classifier_model.optimal_threshold = classifier_model_optimal_threshold
        classifier_model.roc_auc = classifier_model_auc
