import mlflow
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("FaceDetection2")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = f"confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    return cm_path

def generate_classification_report(y_true, y_pred, class_names):
    """Generate and save classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save full report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"classification_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    return report_path, report

def log_model_alternative(model, run_id):
    """Alternative model logging for MLflow versions with endpoint issues"""
    model_path = "tf_model"
    model.save(model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    return f"runs:/{run_id}/model/{model_path}"

with mlflow.start_run() as run:
    # 1. Load model
    model = tf.keras.models.load_model('models/face_recognition_model.h5')
    
    # 2. Data preparation
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # 3. Evaluation
    print("\nEvaluating model...")
    loss, acc = model.evaluate(test_gen)
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    # 4. Performance metrics
    print("Generating performance metrics...")
    cm_path = plot_confusion_matrix(y_true, y_pred, class_names)
    report_path, report = generate_classification_report(y_true, y_pred, class_names)
    
    # 5. Log artifacts
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(report_path)
    
    # 6. Log parameters and metrics
    mlflow.log_params({
        "model_type": "CNN",
        "input_shape": "224x224x3",
        "classes": len(class_names),
        "test_samples": test_gen.samples
    })
    
    mlflow.log_metrics({
        "test_accuracy": acc,
        "test_loss": loss,
        "precision_macro": report['macro avg']['precision'],
        "recall_macro": report['macro avg']['recall'],
        "f1_macro": report['macro avg']['f1-score']
    })
    
    # 7. Save model (using alternative method)
    print("Logging model...")
    model_uri = log_model_alternative(model, run.info.run_id)
    mlflow.log_param("model_uri", model_uri)
    
    # 8. Clean up
    os.remove(cm_path)
    os.remove(report_path)
    
    print(f"\n✅ Evaluation complete! View results at: {mlflow.get_artifact_uri()}")
