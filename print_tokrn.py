# import json

# # Your IDs
# ids = [
#     50258, 50276, 50359, 50363, 3941, 94, 48521, 3941, 94, 48521, 8485, 94, 21981,
#     41858, 17937, 8485, 94, 21981, 41858, 17937, 48449, 8703, 234, 45938, 48521,
#     8485, 255, 21981, 31970, 17937, 8485, 255, 21981, 31970, 17937, 8485, 105,
#     33279, 35082, 8485, 103, 21981, 3941, 237, 48449, 43372, 31945, 8485, 97,
#     33926, 8485, 245, 31881, 25411, 17937, 2031, 19, 37139, 17937, 48268, 46758,
#     27099, 46758, 17937, 49316, 33926, 25411, 36158, 8485, 228, 3941, 103, 41858,
#     31881, 49316, 8703, 223, 3941, 255, 17937, 50257, 50257, 50257, 50257
# ]

# # Load tokenizer.json
# with open("tokenizer_export/tokenizer.json", "r", encoding="utf-8") as f:
#     tok = json.load(f)

# # Reverse vocab
# id_to_token = {v: k for k, v in tok["model"]["vocab"].items()}

# # Convert ids → tokens → text
# tokens = [id_to_token.get(i, "") for i in ids]
# tokens = [t for t in tokens if not t.startswith("<|")]  # drop specials
# text = "".join(tokens)

# print("Raw tokens:", tokens)
# print("Joined text:", text)



# from transformers import WhisperTokenizer

# # Load Whisper tokenizer
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

# # Hindi consonants
# consonants = [
#     "क", "ख", "ग", "घ", "ङ",
#     "च", "छ", "ज", "झ", "ञ",
#     "ट", "ठ", "ड", "ढ", "ण",
#     "त", "थ", "द", "ध", "न",
#     "प", "फ", "ब", "भ", "म",
#     "य", "र", "ल", "व",
#     "श", "ष", "स", "ह",
#     "क्ष", "त्र", "ज्ञ", "श्र"
# ]

# # Hindi vowels (swar)
# vowels = [
#     "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ",
#     "ए", "ऐ", "ओ", "औ",
#     "अं", "अः"
# ]

# # Hindi matras
# matras = ["", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं", "ः"]

# # Output file
# output_file = "hindi_tokens_whisper_full.txt"

# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("### CONSONANTS ###\n")
#     for c in consonants:
#         ids = tokenizer.encode(c, add_special_tokens=False)
#         f.write(f"{c}\t{ids}\n")
    
#     f.write("\n### VOWELS (SWAR) ###\n")
#     for v in vowels:
#         ids = tokenizer.encode(v, add_special_tokens=False)
#         f.write(f"{v}\t{ids}\n")
    
#     f.write("\n### MATRAS ###\n")
#     for m in matras:
#         ids = tokenizer.encode(m, add_special_tokens=False)
#         f.write(f"{m}\t{ids}\n")

# print(f"✅ All consonants, vowels, and matras saved in '{output_file}'")

# from transformers import WhisperProcessor

# processor = WhisperProcessor.from_pretrained("sanchit-gandhi/whisper-small-hi")
# tokenizer = processor.tokenizer

# print(tokenizer.decode([8703]))

from transformers import WhisperProcessor

# Load Hindi Whisper tokenizer
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

# Hindi vowels
vowels = [
    "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ",
    "ए", "ऐ", "ओ", "औ",
    "अं", "अः"
]

# Hindi matras
matras = ["", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ं", "ः", "ॅ", "ॉ", "ॆ", "ॊ", "ॆ", "ॊ", "ॏ", "ॎ", "ॏ", "॑", "॒", "॓", "॔", "ॕ", "ॖ", "ॗ", "ॢ", "ॣ"]

# Function to get both plain + space-prefixed variants
def get_variants(token):
    ids_space = tokenizer.encode(" " + token, add_special_tokens=False)
    ids_plain = tokenizer.encode(token, add_special_tokens=False)
    return ids_space, ids_plain

# Output file
output_file = "hindi_tokens_whisper_full1.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("### CONSONANTS ###\n")
    for c in consonants:
        ids_space, ids_plain = get_variants(c)
        if ids_space:  # with space
            f.write(f" {c}\t{ids_space}\n")
        if ids_plain:  # without space
            f.write(f"{c}\t{ids_plain}\n")
    
    f.write("\n### VOWELS (SWAR) ###\n")
    for v in vowels:
        ids_space, ids_plain = get_variants(v)
        if ids_space:
            f.write(f" {v}\t{ids_space}\n")
        if ids_plain:
            f.write(f"{v}\t{ids_plain}\n")
    
    f.write("\n### MATRAS ###\n")
    for m in matras:
        label = m if m else "[no-matra]"
        ids_space, ids_plain = get_variants(m)
        if ids_space:
            f.write(f" {label}\t{ids_space}\n")
        if ids_plain:
            f.write(f"{label}\t{ids_plain}\n")

print(f"✅ Consonants, vowels, and matras (with & without space) saved in '{output_file}'")
