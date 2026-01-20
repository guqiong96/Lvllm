from transformers import AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained("/home/guqiong/Models/MiniMax-M2.1-AWQ-4bit")

print("Special tokens:")
print(f"  pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"  eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"  bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"  unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

# 检查token_id=0对应什么
print(f"\nToken ID 0 corresponds to: '{tokenizer.decode([0])}'")

# 检查是否存在空字符token
test_tokens = ["", "\x00", "\n", " "]
for t in test_tokens:
    ids = tokenizer.encode(t, add_special_tokens=False)
    print(f"Token '{repr(t)}' -> IDs: {ids}")
 
input_ids = [200034, 200019,  28463,     10,   2985,    457,    258,  12473,  23413,
            46,   5324,   1925,    355,  35353,  12973,   5145,     50,     46,
            49,    306,    355,   6904,    531,  35353,  12973,     46, 200020,
            10, 200019,   3995,     10,  21875,    292,  26901,  22077,    300,
         98058,    296,    292,  88291,     46,    517,  88291,    355,   6692,
            44,    275,  98058,  11779,    296,  14722,    306,  18146,    306,
         21633, 106850,     46,   1781,    566,   3316,   7552,    301,   2391,
           258,   8602,    300,  10888,   3528,    296,  88291,     46,   9771,
         10888,    498,   3075,    258,   3528,   8602,  29232,    301,    412,
            44,   5220,    301,   9458,    412,     46,  10838,    986,    457,
           687,    812,   3528,  11551,     44,  98058,  20993,  31931,    401,
         14975,     46, 200020,     10, 200019,   1361,     10, 200050,     10]  
 
full_text = tokenizer.decode(input_ids)
print(full_text)
 
for i, token_id in enumerate(input_ids[:200]):  
    token_text = tokenizer.decode([token_id])
    print(f"[{i:3d}] ID {token_id:6d}: '{repr(token_text)}'")