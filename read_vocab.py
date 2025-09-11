# import json

# with open("vocab.json", "r", encoding="utf-8") as f:
#     vocab = json.load(f)

# # Show first 50 tokens
# for i, (tok, idx) in enumerate(vocab.items()):
#     print(i, repr(tok), idx)
#     if i > 50:
#         break

# # Search for a known Hindi character
# for tok in vocab.keys():
#     if "म" in tok:
#         print("Found Hindi token:", tok, vocab[tok])
#         break
# from transformers import WhisperTokenizer

# Download Hindi Whisper tokenizer
# tok = WhisperTokenizer.from_pretrained("sanchit-gandhi/whisper-small-hi")

# # Save to local folder
# tok.save_pretrained("./my_tokenizer")

# from transformers import WhisperTokenizerFast

# # Load the fast Hindi tokenizer
# tok = WhisperTokenizerFast.from_pretrained("sanchit-gandhi/whisper-small-hi")

# # Save it, this will include tokenizer.json
# tok.save_pretrained("./my_tokenizer2")
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
tok.sp_model.save("sentencepiece.model")
