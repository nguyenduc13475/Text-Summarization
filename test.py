from tokenization import TransformerTokenizer

a = TransformerTokenizer("vocab.json", "merges.txt")
print(a.id_to_token(49000))
