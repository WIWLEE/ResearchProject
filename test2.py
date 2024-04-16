import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel #tensorflow GPT2 LM
import numpy as np

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
mirrored_strategy = tf.distribute.MirroredStrategy()

#실제 모델에서 제공하는 generate (추론 과정))
sent = '아이들의 정서적인 발달에 동화책은'
input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
with mirrored_strategy.scope():
    output = model.generate(input_ids,
                            batch_size = 10,
                            max_length=200,
                            repetition_penalty=2.0,
                            use_cache=True)

output_ids = output.numpy().tolist()[0]
print(tokenizer.decode(output_ids))
#=====여기부터 워터마킹======