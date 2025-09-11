from transformers import WhisperProcessor
import os

def download_whisper_assets(model_id: str, save_dir: str = "whisper_assets"):
    # Load processor to ensure files are downloaded
    processor = WhisperProcessor.from_pretrained(model_id)

    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save tokenizer files
    processor.tokenizer.save_pretrained(save_dir)

    # 2. Save feature extractor files
    processor.feature_extractor.save_pretrained(save_dir)

    print(f"\n✅ Saved all Whisper assets for '{model_id}' into ./{save_dir}/")
    print("Files:")
    for fname in os.listdir(save_dir):
        print(f"  • {fname}")

if __name__ == "__main__":
    download_whisper_assets("sanchit-gandhi/whisper-small-hi", save_dir="whisper_assets")
