# Step 1: Load your Hindi tokens file
hindi_token_file = "hindi_tokens_whisper_full1.txt"

token_to_char = {}

with open(hindi_token_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        # line = line.strip()
        line = line.rstrip("\n")  # Remove only newline, keep spaces

        if not line or line.startswith("###"):
            continue
        
        # Split into character and IDs
        if "\t" in line:
            char, ids_str = line.split("\t")
        elif " " in line:
            char, ids_str = line.split(" ", 1)
        else:
            continue
        
        # Remove brackets and spaces, then split IDs
        ids_str = ids_str.replace("[", "").replace("]", "").replace(" ", "")
        ids_list = ids_str.split(",")
        for id_str in ids_list:
            if id_str:
                token_to_char[int(id_str)] = char

# --- Add custom mappings for space or other special tokens ---
token_to_char[8485] = " "   # Map 8485 to space
# token_to_char[3941] = " "   # Map visarga

# Step 2: Your list of token IDs to decode
ids = [
    50258, 50276, 50359, 50363, 48521, 8703, 223, 3941, 251, 41858, 33926, 8485, 229, 36158, 35082, 17937, 8485, 105, 36158, 17937, 48268, 17937, 31970, 8703, 234, 3941, 230, 31970, 43372, 45938, 21981, 30, 8485, 97, 27099, 3941, 113, 36158, 49316, 21981, 8485, 99, 33279, 46758, 35082, 17937, 8485, 110, 3941, 245, 17937, 48268, 31970, 8703, 234, 3941, 230, 485, 25411, 3941, 105, 27099, 3941, 105, 17937, 8485, 101, 21981, 8485, 97, 8703, 223, 3941, 250, 41858, 33926, 8485, 105, 35082, 17937, 35082, 21981, 48449, 21981, 31945, 31970, 25411, 8485, 99, 33279, 48268, 17937, 8485, 242, 45938, 35082, 31970, 21981, 8485, 244, 17937, 46758, 31881, 8485, 98, 33279, 3941, 250, 33926, 25411, 33279, 48268, 17937, 31945, 485, 41858, 17937, 3941, 250, 46758, 31970, 21981, 49316, 33279, 48268, 17937, 3941, 237, 49316, 21981, 8485, 110, 33279, 3941, 244, 21981, 8485, 107, 17937, 8485, 97, 33926, 35082, 21981, 8485, 250, 17937, 35082, 21981, 31970, 33279, 36158, 35082, 21981, 485, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257

]

# Step 3: Decode tokens
decoded_text_clean = ""      # Text without token IDs
decoded_text_with_ids = ""   # Text with placeholders for missing tokens
undecoded_tokens = []

for idx, token in enumerate(ids):
    if token in token_to_char:
        decoded_text_clean += token_to_char[token]
        decoded_text_with_ids += token_to_char[token]
    else:
        placeholder = f"[{token}]"
        decoded_text_clean += ""   # Skip in clean version
        decoded_text_with_ids += placeholder
        undecoded_tokens.append((idx, token))

# Step 4: Print results
print("=== Decoded Text (Clean) ===")
print(decoded_text_clean)

print("\n=== Decoded Text (With IDs for missing tokens) ===")
print(decoded_text_with_ids)

print("\n=== Undecoded Tokens (index, token_id) ===")
print(undecoded_tokens)
