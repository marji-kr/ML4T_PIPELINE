# 📁 ml4t_pipeline 폴더 구성 안내

이 문서는 `ml4t_pipeline` 폴더가 어떻게 구성되어 있는지, 각 하위 폴더/파일이 무엇을 위한 것인지 안내합니다. 세부 내용(코드 로직, 산출물 목록)은 각 하위 폴더의 readme 파일([data/data.readme](data/data.readme), [model/model.readme](model/model.readme))을 참고하세요.

---

## 전체 폴더 트리

```
ml4t_pipeline/
├── README.md            # (본 문서) 폴더 구성 안내
├── config.yaml           # 백테스트 파라미터 설정값 (아래 "config.yaml" 항목 참고)
│
├── data/                              # 데이터 수집 및 저장소
│   ├── create_datasets.ipynb          # 원본 데이터를 내려받아 HDF5(assets.h5)로 가공/저장하는 노트북
│   ├── data.readme                    # create_datasets.ipynb 코드 설명 + 저장된 데이터셋 상세 설명
│   ├── assets.h5                      # (구버전/최상위) HDF5 저장소 — 아래 data/data/assets.h5 와는 별개 파일
│   └── data/                          # 원본 소스 파일 + 실제로 노트북들이 참조하는 HDF5 저장소
│       ├── WIKI_PRICES_*.csv          # 미국 개별 종목 과거 주가 원본 (Quandl WIKI Prices)
│       ├── ^uslc_d.csv                # S&P 500 지수 원본 (Stooq)
│       ├── nasdaq.csv / amex.csv / nyse.csv   # 3대 거래소 상장 종목 메타데이터 원본
│       ├── d_jp_txt.zip               # 도쿄증권거래소(TSE) 개별 종목 주가 원본 (model/11단원에서 사용)
│       ├── assets.h5                  # create_datasets.ipynb가 실제로 쓰고, model 노트북들이 읽는 HDF5 저장소
│       ├── japan.h5                   # model/11단원/11단원.ipynb가 d_jp_txt.zip으로부터 생성하는 일본 주가 HDF5
│       ├── mnist/                     # MNIST 손글씨 이미지 데이터 (data, labels .npy)
│       └── fashion_mnist/             # Fashion-MNIST 이미지 데이터 (data, labels .npy, label_dict.csv)
│
└── model/                             # 챕터별 모델링·백테스트 노트북과 산출물
    ├── model.readme                   # 각 챕터(7단원, 11단원) 노트북 코드 설명 + 산출물 파일명 정리 (챕터 학습마다 누적 추가)
    ├── 7단원/                          # Ridge 회귀 기반 다중 호라이즌 예측 + 백테스트
    │   ├── 7단원.ipynb
    │   └── (예측/백테스트 결과 CSV, 모델 .joblib, 그래프 .png 등 — model.readme 참고)
    └── 11단원/                         # Random Forest 기반 다중 호라이즌 예측 + 백테스트 (일본 주식)
        ├── 11단원.ipynb
        └── (예측/백테스트 결과 CSV, 모델 .joblib, 그래프 .png 등 — model.readme 참고)
```

---

## 폴더별 설명

### 1. 루트 (`ml4t_pipeline/`)
- **`README.md`**: 지금 보고 있는 이 문서. 폴더 전체 구성을 안내합니다.
- **`config.yaml`**: `start_date`, `end_date`, `ridge_alpha`, `initial_capital`, `rebalance_period` 값을 담은 설정 파일입니다. 현재 `data/`, `model/` 아래의 노트북들은 이 파일을 직접 읽어오지 않고 각자 노트북 코드 안에 파라미터를 하드코딩해두었으므로, 참고용/이전 버전(과거 `main.py` 기반 파이프라인)의 흔적으로 남아있는 설정 파일입니다.

### 2. `data/` — 데이터 수집 폴더
- **`create_datasets.ipynb`**: Quandl WIKI 주가, FRED/Stooq S&P 500 지수, 위키피디아 S&P 500 구성종목, 3대 거래소 종목 메타데이터, FRED·Yahoo 거시경제 지표, MNIST/Fashion-MNIST를 각각 내려받아 `data/data/assets.h5`(HDF5)와 `data/data/mnist`, `data/data/fashion_mnist` 폴더에 저장합니다.
- **`data.readme`**: 위 노트북이 만드는 각 데이터셋의 기간·변수·용도를 정리한 문서입니다.
- **`data/data/`**: 실제 원본 CSV/ZIP 파일과, 노트북들이 실제로 읽고 쓰는 `assets.h5`가 들어있는 폴더입니다. `model/7단원`, `model/11단원`의 노트북은 모두 `../../data/data/assets.h5` 경로를 통해 이 폴더의 데이터를 읽습니다.
- **`data/assets.h5`** (data 폴더 최상위): `data/data/assets.h5`와 별개의 파일로, 모델 노트북에서는 사용되지 않습니다.

### 3. `model/` — 챕터별 모델링·백테스트 폴더
- 각 하위 폴더(`7단원/`, `11단원/`)는 교재의 해당 단원 내용을 그대로 재현한 주피터 노트북 하나와, 그 노트북을 실행했을 때 생성되는 모든 산출물(예측/성과 CSV, 학습된 모델 `.joblib`, 결과 그래프 `.png`, Zipline 번들용 임시 폴더 `.zipline_root`/`_zipline_csv_bundle`)을 함께 담고 있습니다.
- 산출물 파일명과 그 의미, 어떤 코드 셀에서 생성되는지는 **`model/model.readme`**에 단원별로 정리되어 있으며, 새로운 단원을 학습할 때마다 그 아래에 이어서 추가하는 방식으로 관리합니다.
