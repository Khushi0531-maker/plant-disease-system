"""
CNN Model Architectures for Plant Leaf Disease Classification.
Provides both Transfer Learning (MobileNetV2) and Custom CNN options,
with integrated Data Augmentation layers and Grad-CAM support.
"""

import os

def create_transfer_model(num_classes=38, input_shape=(224, 224, 3), trainable_backbone=False):
    """
    Builds a high-accuracy Transfer Learning CNN using MobileNetV2.
    Ideal for real-time mobile/web inference and rapid convergence.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Data augmentation pipeline embedded directly into the graph
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_flip"),
        layers.RandomRotation(0.15, name="aug_rot"),
        layers.RandomZoom(0.1, name="aug_zoom"),
        layers.RandomContrast(0.1, name="aug_contrast")
    ], name="data_augmentation")

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Preprocessing expected by MobileNetV2: [-1, 1] range
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base pretrained model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_backbone

    # Feature extraction
    features = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantVision_MobileNetV2")
    return model


def create_custom_cnn(num_classes=38, input_shape=(224, 224, 3)):
    """
    Builds an end-to-end custom deep CNN architecture with Conv2D,
    Batch Normalization, MaxPooling, and Dropout.
    Ideal for academic demonstration of CNN fundamentals.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),
        layers.Rescaling(1.0 / 255, name="rescaling"),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),
        layers.Dropout(0.2, name="drop_1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn_2"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
        layers.Dropout(0.25, name="drop_2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        layers.BatchNormalization(name="bn_3"),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv3_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_3"),
        layers.Dropout(0.3, name="drop_3"),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1"),
        layers.BatchNormalization(name="bn_4"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool_4"),
        layers.Dropout(0.35, name="drop_4"),

        # Dense Head
        layers.GlobalAveragePooling2D(name="global_avg_pool"),
        layers.Dense(256, activation="relu", name="dense_fc"),
        layers.Dropout(0.4, name="drop_fc"),
        layers.Dense(num_classes, activation="softmax", name="predictions")
    ], name="PlantVision_CustomCNN")

    return model


def get_or_build_model(weights_path=None, num_classes=38, model_type="mobilenet"):
    """
    Loads weights from file if available, otherwise initializes a ready model.
    """
    import tensorflow as tf

    if model_type == "custom":
        model = create_custom_cnn(num_classes=num_classes)
    else:
        model = create_transfer_model(num_classes=num_classes)

    if weights_path and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded trained weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights directly ({e}). Initializing base architecture.")
            
    return model
