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
id=[50258, 50276, 50359, 50363, 48521, 8703, 223, 3941, 251, 41858, 33926, 8485, 229, 36158, 35082, 17937, 8485, 105, 36158, 17937, 3941, 237, 31970, 33926, 3941, 230, 31970, 43372, 45938, 21981, 8485, 97, 8703, 223, 3941, 250, 45938, 21981, 8485, 99, 33279, 46758, 35082, 17937, 8485, 110, 3941, 245, 17937, 3941, 237, 31970, 33926, 3941, 230, 485, 25411,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257]
# Get output tokens
output_ids = interpreter.get_tensor(output_details[0]['index'])
output_ids = np.array(output_ids).astype(int)
arr =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
# ----------------------------------------------------------
# 5) Decode tokens and save ID->Hindi mapping
# ----------------------------------------------------------
decoded_tokens = processor.batch_decode(output_ids, skip_special_tokens=False)[0]

# Flatten token IDs
flat_ids = output_ids.flatten()

# Save mapping to file
with open("hindi_token_mapping.txt", "w", encoding="utf-8") as f:
    for idx, token_id in enumerate(flat_ids):
        char = decoded_tokens[idx] if idx < len(decoded_tokens) else ""
        f.write(f"{token_id}\t{char}\n")

print(f"TFLite Inference Time: {inference_time_ms:.2f} ms")
print("Output token IDs:", flat_ids.tolist())
print("Predicted text:", processor.batch_decode(output_ids, skip_special_tokens=True)[0])
print("Mapping saved to hindi_token_mapping.txt")
