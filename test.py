self.enc_hidden_to_attn = nn.Linear(encoder_hidden_dim * 2, attention_dim)
self.dec_hidden_to_attn = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)
self.coverage_to_attn = nn.Linear(1, attention_dim, bias=False)
