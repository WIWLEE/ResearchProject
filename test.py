from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
import torch

# 토크나이저와 모델 불러오기
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")

# 프롬프트 정의
prompt = "한국어 자연어 처리 모델에 대한 특정 질문"

# 토크나이징 및 모델 입력 형식으로 변환
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)

# 모델에 입력하여 출력 받기
outputs = model(**inputs)
print(outputs)

# 로짓 벡터 추출
logits = outputs.last_hidden_state
print(logits)
