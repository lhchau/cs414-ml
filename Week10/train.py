from Week10.models.translator import Translator
from Week10.utils.utils import *

num_layers = 2
d_model = 128
dff = 256 
num_heads = 2
dropout_rate = 0.0

alpha=Translator(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=num_eng_tokens,
    target_vocab_size=num_vie_tokens,
    dropout_rate=dropout_rate)

alpha.build()
alpha.summary()