import tensorflow as tf
import numpy as np
from transformers import WhisperProcessor
import time
import soundfile as sf

# ----------------------------------------------------------
# 1) Load tokenizer / processor (for decoding output tokens)
# ----------------------------------------------------------
processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")

# ----------------------------------------------------------
# 2) Load TFLite model
# ----------------------------------------------------------
interpreter = tf.lite.Interpreter(
    model_path="whisper_small_hi_generate.tflite",
    num_threads=8
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------------------------------
# 3) Preprocess audio
# ----------------------------------------------------------
def preprocess_audio(file_path):
    audio, sr = sf.read(file_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="np")
    return inputs.input_features.astype(np.float32)

# ----------------------------------------------------------
# 4) Run inference
# ----------------------------------------------------------
input_features = preprocess_audio("songs/anshal.wav")

interpreter.set_tensor(input_details[0]['index'], input_features)

# Warm-up
interpreter.invoke()

# Timed inference
start = time.time()
interpreter.invoke()
end = time.time()

inference_time_ms = (end - start) * 1000

# Get output tokens
output_ids = interpreter.get_tensor(output_details[0]['index'])

# Make sure they are ints
output_ids = np.array(output_ids).astype(int)

# ----------------------------------------------------------
# 5) Print tokens + decode
# ----------------------------------------------------------
print(f"TFLite Inference Time: {inference_time_ms:.2f} ms")
print("Output token IDs:", output_ids.flatten().tolist())  # <-- raw IDs
print("Predicted text:", processor.batch_decode(output_ids, skip_special_tokens=True)[0])
print(output_details[0])

