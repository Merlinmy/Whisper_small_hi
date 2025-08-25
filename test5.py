import tensorflow as tf
import numpy as np
from transformers import WhisperProcessor
import time

# ----------------------------------------------------------
# 1) Load tokenizer / processor (for decoding output tokens)
# ----------------------------------------------------------
processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")

# ----------------------------------------------------------
# 2) Load TFLite model with multiple CPU threads
#    (TFLite will automatically use XNNPACK if available)
# ----------------------------------------------------------
interpreter = tf.lite.Interpreter(
    model_path="whisper_small_hi_generate.tflite",
    num_threads=8  # <--- change 2, 4, 8 based on your CPU cores
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------------------------------------
# 3) Function to preprocess audio into input_features
# ----------------------------------------------------------
def preprocess_audio(file_path):
    import soundfile as sf
    audio, sr = sf.read(file_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="np")

    # If you want to test float16 (slightly faster), do:
    # return inputs.input_features.astype(np.float16)
    return inputs.input_features.astype(np.float32)


# ----------------------------------------------------------
# 4) Run inference
# ----------------------------------------------------------
input_features = preprocess_audio("songs/kiran.wav")

# Warm-up (first call is slower)
interpreter.set_tensor(input_details[0]['index'], input_features)
interpreter.invoke()

# Timed call
start = time.time()
interpreter.invoke()
end = time.time()
inference_time_ms = (end - start) * 1000

# Get output
output_ids = interpreter.get_tensor(output_details[0]['index'])

# Decode tokens to text
text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print(f"TFLite Inference Time: {inference_time_ms:.2f} ms")
print("Predicted text:", text)
