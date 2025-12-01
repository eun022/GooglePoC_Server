# from fastapi import FastAPI
# import logging
# from os import path, environ
# from dataclasses import dataclass, asdict
# from starlette.config import Config as E_config
# import requests
# from PIL import Image
# from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, GenerationConfig
# import torch
# import os
# import threading

# base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
# base_ai_dir = "ai/models"
# _init_lock = threading.Lock()

# class MultiModel:
#     def __init__(self, app : FastAPI = None, **kwargs):
#         self._inited = False
#         self._vlm = None
#         # self._translator = None
            

#         if app is not None:
#             self.init_app(app=app, **kwargs)

#     def init_app(self, app : FastAPI, **kwargs):
#         """
#         :param app:  FastAPI 인스턴스
#         :param kwargs: config 파일 설정
#         :return:
#         """
#         if self._inited:
#             return  # ✅ 중복 방지
#         with _init_lock:
#             if self._inited:
#                 return

#             model_path = f"./{base_ai_dir}/sa2va/axis_data_ele"
#             vlm_model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 torch_dtype="auto",
#                 device_map={"": "cuda:0"}, #
#                 trust_remote_code=True
#             ).eval()

#             vlm_tokenizer = AutoTokenizer.from_pretrained(
#                             model_path,
#                             trust_remote_code=True
#                         )

#             self._vlm = {"model" : vlm_model, "token" : vlm_tokenizer}

#             # t5_name = environ.get("VXFZ_TRANSLATOR_MODEL_T5_NAME", "azaraks/t5-xlarge-ko-kb")
#             # max_len = int(os.environ.get("VXFZ_TRANSALTOR_MODEL_T5_MAX_LENGTH", "256")) # env 키 오탈자 주의
#             # max_in  = max_len >> 2

#             # t5_model = T5ForConditionalGeneration.from_pretrained(
#             #     t5_name,
#             #     torch_dtype= torch.bfloat16,
#             #     device_map="auto"
#             # ).eval()
#             # t5_tokenizer = AutoTokenizer.from_pretrained(t5_name)
#             # gen_cfg = GenerationConfig(max_length=max_len, num_beams=4, early_stopping=True)

#             # self._translator = {
#             #     "model": t5_model,
#             #     "tokenizer": t5_tokenizer,
#             #     "max_length": max_len,
#             #     "max_input_length": max_in,
#             #     "gen_cfg": gen_cfg
#             # }
#             self._inited = True

#     @property
#     def vlm(self):
#         return self._vlm

#     # @property
#     # def translator(self):
#     #     return self._translator
        

#     def _split_by_tokenizer(self, text: str):
#         assert self._translator is not None, "Translator is not initialized"
#         tokenizer = self._translator["tokenizer"]
#         max_len = self._translator["max_length"]
#         max_in  = self._translator["max_input_length"]

#         words = text.split()
#         result, current = [], []
#         for w in words:
#             current.append(w)
#             chunk = " ".join(current)
#             tokens = tokenizer(chunk, return_tensors="pt", max_length=max_len, truncation=True)
#             if tokens.input_ids.shape[1] > max_in:
#                 current.pop()
#                 if current:
#                     result.append(" ".join(current))
#                 current = [w]
#         if current:
#             result.append(" ".join(current))
#         return result
#     @staticmethod
#     def _infer_device_from_model(model) -> torch.device:
#     # sharded 모델에도 안전: 첫 파라미터의 디바이스
#         try:
#             return next(model.parameters()).device
#         except StopIteration:
#             # 가끔 파라미터가 meta일 수 있음 → cpu로 폴백
#             return torch.device("cpu")
#     @staticmethod
#     def _to_model_device(batch, model):
#         dev = MultiModel._infer_device_from_model(model)
#         return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}


#     # ---- 여기서부터 추가: T5 번역 실행 ----
#     # @torch.inference_mode()
#     # def translate(self, text: str) -> str:
        
#     #     assert self._translator is not None, "Translator is not initialized"
#     #     model = self._translator["model"]
#     #     tokenizer = self._translator["tokenizer"]
#     #     max_len = self._translator["max_length"]
  
    
#     #     sanitized_text = text.replace("_", " ")
#     #     chunks = self._split_by_tokenizer(sanitized_text)
#     #     outs = []
#     #     for ch in chunks:
#     #         instruction = f'translate Korean to Braille: "{ch}"'
#     #         #print(instruction)
#     #         inputs = tokenizer(
#     #             instruction,
#     #             return_tensors="pt",
#     #             max_length=max_len,
#     #             truncation=True,
#     #         )
#     #         # device_map="auto" 사용 시: 모델의 주 디바이스로 이동
#     #         inputs = self._to_model_device(inputs, model)
#     #         gen = model.generate(**inputs,
#     #                              max_length=max_len,
#     #                              num_beams=4,
#     #                              early_stopping=True)
            
#     #         #print(gen)
#     #         text_out = tokenizer.decode(gen[0], skip_special_tokens=False)
#     #         #print("여기가 모델 안 text_out",text_out)
#     #         # 따옴표만 깔끔히 제거(하드코딩 슬라이싱 제거)
#     #         outs.append(text_out[6:-5])
#     #         output_text = ' '.join(outs)
#     #         print("점자 변환",text, outEput_text)

#     #     return output_text

# mainModel = MultiModel()

