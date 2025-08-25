import tensorflow as tf
from transformers import TFWhisperForConditionalGeneration

# 1. Load your Hugging Face model (this will instantiate the subclassed model)
model = TFWhisperForConditionalGeneration.from_pretrained("sanchit-gandhi/whisper-small-hi")

# 2. Build the model with a dummy input shape
#    This step is crucial to create the model's variables
dummy_input_shape = (1, 80, 3000)  # Example: Adjust according to your model's actual input shape
model.build(dummy_input_shape)

# 3. Load the .h5 weights
model.load_weights("tf_model.h5", by_name=True, skip_mismatch=True) 

