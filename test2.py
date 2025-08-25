import tensorflow as tf

saved_model_dir = "saved_model"  # path to your SavedModel directory
tflite_model_path = "whisper_small_hi.tflite"

# Create converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Enable TensorFlow ops not in TFLite
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved at", tflite_model_path)
