import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel #tensorflow GPT2 LM
import numpy as np

model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
mirrored_strategy = tf.distribute.MirroredStrategy()

sent = '아이들의 정서적인 발달에 동화책은'

input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
print(input_ids)

output = model(input_ids)
Logits = output.logits

with mirrored_strategy.scope():
    while input_ids.shape[-1] < 100:
     
        print("================================")
        print(input_ids.shape[-1])
        Logits = model(input_ids).logits
        
        add_token_id = tf.argmax(Logits)
        add_token_id = tf.reshape(add_token_id, [1,1])
        add_token_id = tf.cast(add_token_id, input_ids.dtype)
        input_ids = tf.concat([input_ids, add_token_id], axis=-1)
        
        next_token = tokenizer.decode(input_ids.numpy()[0,:])
        print(next_token)
    
        


last_index = input_ids.shape[-1]

with mirrored_strategy.scope():
    while input_ids.shape[-1] < 100:
        print("================================")
        print(input_ids.shape[-1])
        #set 나눠서 green set 로짓을 촉진하기
        gamma = 1.0
        delta = 0.0
        Logits = model(input_ids).logits
        print(f"original Logits : {Logits}")

        #seed = int(hash(input_ids % 51200))
        seed = input_ids[-1] % Logits.shape[-1]
        print(f"seed : {seed[-1]}")
        np.random.seed(seed[-1])

        indices = np.arange(Logits.shape[-1]) #0~51999
        green_list_size = int(Logits.shape[-1] * gamma)  #25600
        green_list = np.random.choice(indices, green_list_size, replace=False)
        print(f"green list : {green_list} {green_list.shape}") #25600개의 green list

        #그린 리스트에 속한 idx 값의 Logit에는 델타 추가!!!!!!
        # 인덱스와 업데이트를 위한 텐서 생성
        print(f"last index = last words : {last_index}")
        
        #--여기부터 probability 구하기
        
        green_logits = tf.zeros(green_list_size)
        red_logits = tf.zeros(Logits.shape[-1] - green_list_size)
        for i in green_list:
            indices = [[0, last_index - 1, i]]  # 차원에 맞는 인덱스 설정
            updates = [Logits[0, last_index - 1, i] + delta]  # 업데이트할 값
            Logits = tf.tensor_scatter_nd_update(Logits, indices, updates)

        print(f"updated Logits : {Logits}")
        
        denominator = tf.reduce_sum(tf.exp(Logits), axis=-1)  # softmax 계산을 위한 분모
        print(f"dominants : {denominator}") #25600개의 green list

        # Logits 텐서에 softmax 적용하여 확률 계산
        softmax_tensor = tf.nn.softmax(Logits[0, last_index - 1, :])
        print(softmax_tensor)
        print(softmax_tensor.shape)


        print(f"make Probabilities : {softmax_tensor}")

        # 데이터 타입 확인 및 조정
        add_token_id = tf.argmax(softmax_tensor)
        print(f"updated Logits Max value : {add_token_id}")


        # 텐서 연결
        add_token_id = tf.reshape(add_token_id, [1,1])
        add_token_id = tf.cast(add_token_id, input_ids.dtype)
        input_ids = tf.concat([input_ids, add_token_id], axis=-1)
        next_token = tokenizer.decode(input_ids.numpy()[0,:])
        print(f"next token : {next_token}")
        
        last_index = input_ids.shape[-1]