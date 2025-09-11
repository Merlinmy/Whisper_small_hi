from transformers import WhisperProcessor

# Load the Hindi Whisper processor
processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")
tokenizer = processor.tokenizer

# Hindi consonants
consonants = [
    "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "क्ष", "त्र", "ज्ञ", "श्र"
]

# Hindi vowels (swar)
vowels = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ",
    "ए", "ऐ", "ओ", "औ",
    "अं", "अः"
]

# Hindi matras
matras = ["", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं", "ः"]

# Output file
output_file = "hindi_letters_tokens.txt"

with open(output_file, "w", encoding="utf-8") as f:
    # Consonants
    f.write("### CONSONANTS ###\n")
    for c in consonants:
        ids = tokenizer.encode(c, add_special_tokens=False)
        f.write(f"{c}\t{ids}\n")
    
    # Vowels
    f.write("\n### VOWELS (SWAR) ###\n")
    for v in vowels:
        ids = tokenizer.encode(v, add_special_tokens=False)
        f.write(f"{v}\t{ids}\n")
    
    # Matras
    f.write("\n### MATRAS ###\n")
    for m in matras:
        ids = tokenizer.encode(m, add_special_tokens=False)
        f.write(f"{m}\t{ids}\n")

print(f"✅ Hindi consonants, vowels, and matras with their token IDs saved in '{output_file}'")
