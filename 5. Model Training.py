# -*- coding: utf-8 -*-
"""Optimized Plant Detection with GPU Handling.ipynb"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import drive
import os
import gc
from sklearn.metrics import classification_report

# ======================
# GPU Configuration
# ======================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable optimizations only if GPU exists
        tf.config.optimizer.set_jit(True)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU config error: {e}")
else:
    print("⚠️ No GPU detected - Using CPU")

# ======================
# Drive Setup
# ======================
drive.mount('/content/drive', force_remount=True)
BASE_PATH = '/content/drive/MyDrive/plant anomaly detection'
SAVE_PATH = os.path.join(BASE_PATH, 'saved_models')
os.makedirs(SAVE_PATH, exist_ok=True)

# ======================
# Constants
# ======================
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16 if gpus else 8
NUM_CLASSES = 4
EPOCHS = 20
SEED = 42

# ======================
# Data Pipeline (Fixed)
# ======================
def create_data_flow(subset):
    generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=35,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    flow = generator.flow_from_directory(
        os.path.join(BASE_PATH, subset),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=(subset == 'train'),
        seed=SEED
    )
    
    # Convert to tf.data.Dataset for performance optimizations
    dataset = tf.data.Dataset.from_generator(
        lambda: flow,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *IMAGE_SIZE, 3], [None, NUM_CLASSES])
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE).cache()

print("\n🔄 Creating data flows...")
train_ds = create_data_flow('train')
val_ds = create_data_flow('val')
test_ds = create_data_flow('test')

# ======================
# Model Architecture
# ======================
def build_model(base_model):
    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', dtype='float32')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ======================
# Model Initialization
# ======================
MODELS = {
    'EfficientNetB7': applications.EfficientNetB7,
    'Xception': applications.Xception,
    'InceptionResNetV2': applications.InceptionResNetV2,
    'DenseNet201': applications.DenseNet201,
    'ResNet152V2': applications.ResNet152V2,
    'NASNetLarge': applications.NASNetLarge
}

print("\n🧠 Initializing models...")
models = []
for name, creator in MODELS.items():
    try:
        print(f"⚙️ Building {name}...")
        base = creator(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        base.trainable = False
        model = build_model(base)
        models.append((name, model))
        print(f"✅ {name} initialized | Params: {model.count_params()/1e6:.1f}M")
    except Exception as e:
        print(f"❌ {name} failed: {str(e)}")

# ======================
# Training Setup
# ======================
callbacks = [
    callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        os.path.join(SAVE_PATH, '{epoch}_{val_accuracy:.2f}.h5'),
        save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

# ======================
# Training Loop
# ======================
print("\n🚀 Starting training...")
for name, model in models:
    print(f"\n{'='*60}")
    print(f"🔥 Training {name}")
    print(f"{'='*60}")
    
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(os.path.join(SAVE_PATH, f'{name}_final.h5'))
        print(f"💾 Saved {name} to Drive")
        
    except Exception as e:
        print(f"❌ Training failed for {name}: {str(e)}")
    
    # Memory cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# ======================
# Ensemble Evaluation
# ======================
print("\n📊 Generating ensemble predictions...")
predictions = []

for name, _ in models:
    try:
        print(f"🔮 Loading {name}...")
        model = tf.keras.models.load_model(os.path.join(SAVE_PATH, f'{name}_final.h5'))
        preds = model.predict(test_ds, verbose=0)
        predictions.append(preds)
        print(f"✅ {name} predictions complete")
        
        del model
        gc.collect()
    except Exception as e:
        print(f"❌ Failed to load {name}: {str(e)}")

# Final evaluation
ensemble_preds = np.mean(predictions, axis=0)
final_labels = np.argmax(ensemble_preds, axis=1)

print("\n📈 Final Classification Report:")
print(classification_report(test_ds.classes, final_labels, target_names=train_ds.class_indices.keys()))

print("\n🎉 All operations completed! Models saved in:", SAVE_PATH)
