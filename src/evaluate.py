from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, val):
    val_preds = model.predict(val)
    y_pred = np.argmax(val_preds, axis=1)
    y_true = val.classes

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d',
                xticklabels=val.class_indices.keys(),
                yticklabels=val.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=val.class_indices.keys()))

    # Accuracy
    val_loss, val_acc = model.evaluate(val)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
