from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from classes.ClassifierMetricsVisualizer import ClassifierMetricsVisualizer
from classes.ClassifierModel import ClassifierModel
from classes.ClassifierProcessor import ClassifierProcessor
from classes.DataProcessor import DataProcessor


def main():
    # 1. Download bioresponse.csv file
    x_data, y_data = DataProcessor.parse_csv_by_attribute("data/bioresponse.csv", "Activity")
    x_train_data, x_test_data, y_train_data, y_test_data = DataProcessor.split_by_train_and_test_data(x_data, y_data)

    # 2. Train 4 classifiers
    dt_classifier_model = ClassifierModel(DecisionTreeClassifier(), "Decision Tree")
    rf_classifier_model = ClassifierModel(RandomForestClassifier(), "Random Forest")
    dt_classifier_deep_model = ClassifierModel(DecisionTreeClassifier(max_depth=10), "Deep Decision Tree")
    rf_classifier_deep_model = ClassifierModel(RandomForestClassifier(max_depth=10), "Deep Random Forest")

    models = [dt_classifier_model, rf_classifier_model, dt_classifier_deep_model, rf_classifier_deep_model]

    for classifier_model in models:
        classifier_model.classifier.fit(x_train_data, y_train_data)

    # 3. Calculate accuracy, precision, recall, f1 score and log loss
    for classifier_model in models:
        ClassifierProcessor.evaluate_classifier(classifier_model, x_test_data, y_test_data)

    # 4. Plot precision-recall and ROC-curves
    for classifier_model in models:
        ClassifierProcessor.calculate_precisions_recalls_and_roc_curves_and_optimal_threshold(classifier_model,
                                                                                              x_test_data, y_test_data)
        ClassifierMetricsVisualizer.plot_precisions_recalls_and_roc_curves(classifier_model)

    # Print 4 classifiers properties
    for classifier_model in models:
        print(classifier_model)

    # 5.1 Train avoids type II errors classifier
    avoid_type_ii_error_rf_classifier_model = ClassifierModel(RandomForestClassifier(),
                                                              "Avoid Type II Error Random Forest")
    avoid_type_ii_error_rf_classifier_model.classifier.fit(x_train_data, y_train_data)

    # 5.2 Calculate quality metrics
    ClassifierProcessor.calculate_precisions_recalls_and_roc_curves_and_optimal_threshold(
        avoid_type_ii_error_rf_classifier_model, x_test_data, y_test_data)
    ClassifierMetricsVisualizer.plot_precisions_recalls_and_roc_curves(avoid_type_ii_error_rf_classifier_model)

    ClassifierProcessor.evaluate_classifier(avoid_type_ii_error_rf_classifier_model, x_test_data, y_test_data, True)
    print(avoid_type_ii_error_rf_classifier_model)


if __name__ == "__main__":
    main()
