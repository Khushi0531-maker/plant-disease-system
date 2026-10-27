"""
Standalone Training Pipeline for Plant Disease Detection CNN.
Supports Transfer Learning (MobileNetV2) and Custom CNN, with automatic
data augmentation, learning rate scheduling, early stopping, and performance curves.
"""

import os
import argparse
import matplotlib.pyplot as plt
from disease_info import CLASS_NAMES
from model_builder import create_transfer_model, create_custom_cnn

def train_model(
    data_dir,
    model_type="mobilenet",
    epochs=15,
    batch_size=32,
    img_size=(224, 224),
    learning_rate=1e-3,
    save_path="plant_disease_model.keras"
):
    import tensorflow as tf

    print(f"\n🌱 Initializing Training on {data_dir} using {model_type.upper()}...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    # 1. Ingest Dataset with 80% train / 20% validation split
    print("Loading train and validation datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    detected_classes = train_ds.class_names
    num_classes = len(detected_classes)
    print(f"Detected {num_classes} classes: {detected_classes}")

    # Optimize dataset loading with prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 2. Build Selected CNN Architecture
    if model_type.lower() == "custom":
        model = create_custom_cnn(num_classes=num_classes, input_shape=(*img_size, 3))
    else:
        model = create_transfer_model(num_classes=num_classes, input_shape=(*img_size, 3), trainable_backbone=False)

    model.summary()

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    # 4. Callbacks for Best Accuracy & Overfitting Prevention
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Model Training
    print(f"\n🚀 Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # 6. Save Final Training Plots
    plot_training_results(history, output_file="training_curves.png")
    print(f"\n✅ Training complete! Best weights saved to: {save_path}")
    print("📈 Training accuracy/loss curves saved to: training_curves.png")

    return model, history


def plot_training_results(history, output_file="training_curves.png"):
    """Plots and exports loss and accuracy curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs_range, acc, 'o-', color='#16a34a', label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, 's--', color='#0284c7', label='Validation Accuracy')
    ax1.set_title('CNN Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Loss plot
    ax2.plot(epochs_range, loss, 'o-', color='#dc2626', label='Training Loss')
    ax2.plot(epochs_range, val_loss, 's--', color='#ea580c', label='Validation Loss')
    ax2.set_title('Cross-Entropy Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Plant Disease Detection")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset directory structured by class folders")
    parser.add_argument("--model_type", type=str, default="mobilenet", choices=["mobilenet", "custom"], help="CNN architecture type")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--save_path", type=str, default="plant_disease_model.keras", help="Target model weights filename")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
