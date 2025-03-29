import matplotlib.pyplot as plt
import numpy as np


class ClassifierMetricsVisualizer:

    @staticmethod
    def plot_precisions_recalls_and_roc_curves(classifier_model):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(classifier_model.recalls, classifier_model.precisions)
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_title(f"Precision-Recall Curve ({classifier_model.name})")

        optimal_threshold_index = np.argmax(classifier_model.tp_rates - classifier_model.fp_rates)

        axes[1].plot(classifier_model.fp_rates, classifier_model.tp_rates,
                     label=f"ROC Curve (AUC = {classifier_model.roc_auc:.2f})")
        axes[1].scatter(classifier_model.fp_rates[optimal_threshold_index],
                        classifier_model.tp_rates[optimal_threshold_index], marker="o", color="r",
                        label="Optimal Threshold")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(f"ROC Curve ({classifier_model.name})")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
