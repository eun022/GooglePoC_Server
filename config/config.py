from os import path, environ
from dataclasses import dataclass, asdict
from starlette.config import Config as E_config



config = E_config('.env')
base_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))

@dataclass
class Config:

    BASE_DIR: str = base_dir
    DB_POOL_RECYCLE: int = 900
    DB_ECHO: bool = True
    # EDGE_MODELS : str =  "edge_detection_fq.onnx"
    # SAM_MODELS1 : str =  "vit_b_encoder.onnx"
    # SAM_MODELS2 : str =  "vit_b_decoder.onnx"
    # YOLO_MODELS : str =  "yolo11s-seg.onnx"
    #PATHS : str = "./ai/models"

# @dataclass
# class LocalConfig(Config):
#     """
#     Local Config 파일 추가하기
#     """
#     PROJ_RELOAD : bool = False
#     DB_URL:str =f"sqlite:///{base_dir}/myapi.db"
#     ALLOW_SITE = ["*"]
#     TRUSTED_HOSTS = ["*"]
#     API : str = config("OPENAPI_KEY")
    


# @dataclass
# class ProdConfig(Config):
#     """
#     Prod Config 파일 추가하기
#     """
#     PROJ_RELOAD: bool = True
#     ALLOW_SITE = ["*"]
#     TRUSTED_HOSTS = ["*"]

# def conf():
#     """
#     환경 불러오기
#     :return:
#     """
#     config = dict(prod=ProdConfig() , local=LocalConfig())
#     print(config)
#     return config.get(environ.get("API_ENV", "local"))



@dataclass
class Config:
    BASE_DIR: str = base_dir
    PROJ_RELOAD: bool = False
    ALLOW_SITE: list = ["*"]
    TRUSTED_HOSTS: list = ["*"]
    MODEL_PATH: str = f"{base_dir}/ai/models/sa2va/axis_data_ele"

# 그냥 바로 이 Config 인스턴스를 사용하면 됩니다.
def conf():
    return Config()