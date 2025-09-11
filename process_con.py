from transformers import WhisperProcessor
import os
import json

# ----------------------------------------------------------
# Load processor
# ----------------------------------------------------------
model_id = "sanchit-gandhi/whisper-small-hi"
processor = WhisperProcessor.from_pretrained(model_id)

# ----------------------------------------------------------
# Inspect components
# ----------------------------------------------------------
print("=== Whisper Preprocessor Components ===")
print("Tokenizer class:", processor.tokenizer.__class__.__name__)
print("Feature extractor class:", processor.feature_extractor.__class__.__name__)

# Tokenizer details
print("\n=== Tokenizer Info ===")
print("Vocab size:", processor.tokenizer.vocab_size)
print("Special tokens:", processor.tokenizer.special_tokens_map)
print("Tokenizer files location:", processor.tokenizer.init_kwargs.get("name_or_path", "unknown"))

# Feature extractor details
print("\n=== Feature Extractor Info ===")
print("Sampling rate:", processor.feature_extractor.sampling_rate)
print("Hop length:", processor.feature_extractor.hop_length)
print("Chunk length (s):", processor.feature_extractor.chunk_length)

# ----------------------------------------------------------
# Save actual files used
# ----------------------------------------------------------
save_dir = "./whisper_assets"
os.makedirs(save_dir, exist_ok=True)

processor.tokenizer.save_pretrained(save_dir)
processor.feature_extractor.save_pretrained(save_dir)

print(f"\n✅ All tokenizer + feature extractor files saved to: {save_dir}")

# Optional: list saved files
print("\nSaved files:")
for f in os.listdir(save_dir):
    print(" -", f)

# ----------------------------------------------------------
# Peek into normalizer.json if available
# ----------------------------------------------------------
normalizer_path = os.path.join(save_dir, "normalizer.json")
if os.path.exists(normalizer_path):
    with open(normalizer_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    print("\n=== Normalizer Rules (sample) ===")
    print(json.dumps(rules, indent=2)[:1000])  # print first 1000 chars
else:
    print("\n(no normalizer.json found for this model)")
