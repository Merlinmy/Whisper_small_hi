from huggingface_hub import snapshot_download

# Downloads to a local folder
snapshot_download("sanchit-gandhi/whisper-small-hi", cache_dir="./whisper_hi_assets")
