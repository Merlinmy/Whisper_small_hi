import tensorflow as tf
import numpy as np
from transformers import WhisperProcessor
import time

# Load processor for token decoding
processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="whisper_small_hi_generate.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess audio
def preprocess_audio(file_path):
    import soundfile as sf
    audio, sr = sf.read(file_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="np")
    return inputs.input_features.astype(np.float32)

# Preprocess wav
input_features = preprocess_audio("songs/kiran.wav")

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_features)
start = time.time()
interpreter.invoke()
end = time.time()
inference_time_sec = end - start
inference_time_ms = (end - start) * 1000
print(f"TFLite inference time: {inference_time_sec:.4f} sec")
output_ids = interpreter.get_tensor(output_details[0]['index'])

# Decode tokens to text
text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print("Predicted text:", text)
print(f"TFLite Inference time : {inference_time_ms:.2f} ms")


# tested teh model and go tthe results