# Step 1: Load Hindi tokens mapping
hindi_token_file = "hindi_tokens_whisper_full1.txt"

# Instead of single ID mapping, we’ll store tuple-of-IDs -> char
seq_to_char = {}

with open(hindi_token_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip("\n")
        if not line or line.startswith("###"):
            continue

        if "\t" in line:
            char, ids_str = line.split("\t")
        elif " " in line:
            char, ids_str = line.split(" ", 1)
        else:
            continue

        ids_str = ids_str.replace("[", "").replace("]", "").replace(" ", "")
        ids_list = [int(x) for x in ids_str.split(",") if x]

        # store the entire tuple as a key
        seq_to_char[tuple(ids_list)] = char

# Add custom overrides if needed
seq_to_char[(8485,)] = " "   # example: space
seq_to_char[(5027,)] = "<EOT>"  # end-of-text marker

# Step 2: Token IDs to decode
ids = [
   50258, 50276, 50359, 50363, 48521, 8703, 223, 3941, 251, 41858, 33926, 8485, 229, 36158, 35082, 17937, 8485, 105, 36158, 17937, 48268, 17937, 31970, 8703, 234, 3941, 237, 31970, 43372, 45938, 21981, 30, 8485, 97, 27099, 3941, 113, 36158, 45938, 21981, 8485, 99, 33279, 46758, 35082, 17937, 8485, 110, 3941, 245, 17937, 48268, 31970, 8703, 234, 3941, 237, 485, 25411, 3941, 105, 27099, 3941, 105, 17937, 8485, 101, 21981, 8485, 97, 8703, 223, 3941, 250, 41858, 33926, 8485, 105, 35082, 17937, 35082, 21981, 48449, 21981, 31945, 31970, 25411, 8485, 99, 33279, 48268, 17937, 8485, 242, 45938, 35082, 31970, 21981, 8485, 244, 17937, 46758, 31881, 8485, 98, 33279, 3941, 250, 33926, 25411, 33279, 48268, 17937, 31945, 485, 41858, 17937, 3941, 250, 46758, 31970, 21981, 49316, 33279, 48268, 17937, 3941, 237, 49316, 21981, 8485, 110, 33279, 3941, 244, 21981, 8485, 107, 44500, 8485, 97, 33926, 35082, 21981, 8485, 250, 17937, 35082, 21981, 31970, 33279, 36158, 35082, 21981, 485, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257, 50257
]
# Step 3: Greedy decode
decoded_text_clean = ""
decoded_text_with_ids = ""
undecoded_tokens = []
mapping_log = []  # store logs for writing to output.txt

i = 0
while i < len(ids):
    matched = False
    # Try longest first: 4 → 3 → 2 → 1
    for length in [4, 3, 2, 1]:
        if i + length <= len(ids):
            seq = tuple(ids[i:i+length])
            if seq in seq_to_char:
                char = seq_to_char[seq]
                decoded_text_clean += char
                decoded_text_with_ids += char
                mapping_log.append(f"{seq} -> {char}")
                i += length
                matched = True
                break
    if not matched:
        # no match, keep token as ID
        token = ids[i]
        decoded_text_with_ids += f"[{token}]"
        decoded_text_clean += ""  # skip in clean version
        undecoded_tokens.append((i, token))
        mapping_log.append(f"({token},) -> [UNMAPPED]")
        i += 1

# Step 4: Print results
print("=== Decoded Text (Clean) ===")
print(decoded_text_clean)

print("\n=== Decoded Text (With IDs for missing tokens) ===")
print(decoded_text_with_ids)

print("\n=== Undecoded Tokens (index, token_id) ===")
print(undecoded_tokens)

# Step 5: Save mapping logs to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Token Mapping Log:\n")
    for log in mapping_log:
        f.write(log + "\n")

print("\n✅ Token-to-char mapping log saved to output.txt")
