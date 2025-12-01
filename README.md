# 📌 DotStem API Server

본 프로젝트는 DotStem 서비스의 API 서버 환경을 구성하고 실행하기 위한 가이드입니다.  
아래 순서를 따라 하면 서버를 바로 실행할 수 있습니다.

---

## 🧪 1) Conda 가상환경 생성

```bash
conda create -n DotStem python=3.10.18
conda activate DotStem

## 📦 2) 패키지 설치

`requirements2.txt` 기준으로 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements2.txt
```

##  🖼 3) Matplotlib 한글 폰트 설치
```bash
sudo apt-get update
sudo apt-get install -y fonts-noto-cjk
```
##  🔐 4) 환경변수(.env) 설정

DotStem-api/.env 파일을 생성하고 아래 내용을 채워 넣습니다:

GEMINI_API_KEY=


※ 필요한 Key는 별도로 발급받아 입력해야 합니다.

# 🧹 기타 (선택)

## 📁 static 데이터 초기화
다음 명령으로 static 폴더 내 모든 파일을 삭제할 수 있습니다:
```bash
find static -type f -delete
```
##  🟢 실행

프로젝트 루트(DotStem-api)에서:

```bash
python main.py
```