class ClassifierModel:

    def __init__(self, classifier, name: str, accuracy: float = 0.0, precision: float = 0.0, recall: float = 0.0,
                 f1: float = 0.0, loss: float = 0.0, precisions=None, recalls=None, fp_rates=None, tp_rates=None,
                 optimal_threshold: float = 0.0, roc_auc: float = 0.0):
        self.classifier = classifier
        self.name = name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.loss = loss

        if precisions is None:
            precisions = []
        if recalls is None:
            recalls = []
        if tp_rates is None:
            tp_rates = []
        if fp_rates is None:
            fp_rates = []

        self.precisions = precisions
        self.recalls = recalls
        self.fp_rates = fp_rates
        self.tp_rates = tp_rates
        self.optimal_threshold = optimal_threshold
        self.roc_auc = roc_auc

    def __str__(self):
        return (
            f"{self.name} Classifier Model Metrics:\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1 Score: {self.f1:.4f}\n"
            f"  Loss: {self.loss:.4f}\n"
            f"  Optimal Threshold: {self.optimal_threshold:.4f}\n"
            f"  AUC: {self.roc_auc:.4f}\n"
        )
