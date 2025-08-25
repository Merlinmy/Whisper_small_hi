# import tensorflow as tf

# saved_model_dir = "saved_model"  # Your original TF SavedModel
# fixed_model_dir = "saved_model_fixed"
# tflite_model_path = "whisper_small_hi_fixed.tflite"

# # Load the model
# model = tf.saved_model.load(saved_model_dir)

# # Create a concrete function with fixed input shape for Whisper
# @tf.function(input_signature=[tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)])
# def serve(input_tensor):
#     return model(input_tensor)

# # Save with the new signature
# tf.saved_model.save(model, fixed_model_dir, signatures={"serving_default": serve})

# # Convert to TFLite
# converter = tf.lite.TFLiteConverter.from_saved_model(fixed_model_dir)
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tflite_model = converter.convert()

# # Save TFLite model
# with open(tflite_model_path, "wb") as f:
#     f.write(tflite_model)

# print(f"✅ Fixed-shape TFLite model saved at {tflite_model_path}")




import tensorflow as tf
from transformers import TFWhisperForConditionalGeneration, WhisperProcessor

# 1️⃣ Load model & processor
model_id = "sanchit-gandhi/whisper-small-hi"
model = TFWhisperForConditionalGeneration.from_pretrained(model_id, from_pt=True)
processor = WhisperProcessor.from_pretrained(model_id)

# 2️⃣ Create a serving function for full generate()
@tf.function(input_signature=[tf.TensorSpec([1, 80, 3000], tf.float32, name="input_features")])
def generate_from_features(input_features):
    generated_ids = model.generate(input_features, max_length=225)
    return {"output_ids": generated_ids}

# 3️⃣ Save as SavedModel with only one input & one output
saved_path = "whisper_generate_saved_model"
tf.saved_model.save(model, saved_path, signatures={"serving_default": generate_from_features})

# 4️⃣ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
tflite_model = converter.convert()

with open("whisper_small_hi_generate.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved at whisper_small_hi_generate.tflite")



# Got the model by this code