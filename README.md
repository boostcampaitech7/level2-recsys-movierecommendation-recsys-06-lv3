# Movie Recommendation 대회 프로젝트

## 프로젝트 구조

<details>
    <summary> 프로젝트 구조</summary>

```bash
movie_recommendation/
├── config/
│   ├── config-sample.yaml # 모델 선택 및 데이터 , wandb 등
│   └── model_configs/  # recbole 관련 모델 설정 yaml
│       ├── gru4rec.yaml
│       ├── sasrec.yaml
│       ├── bert4rec.yaml
│       ├── lightgcn.yaml
│       └──  fm.yaml
│
├── data/
│   ├── train/ # train 데이터
│   ├── recbole/ # 데이터셋 처리후 recbole 데이터셋화
│   └── eval/
│
├── src/
│   ├── data/ # 데이터셋
│   │   ├── __init__.py # 사용 가능한 dataset class 집합
│   │   ├── DataInterface.py # 데이터셋 관련 Interface
│   │   └── *.py # ~ 관련 데이터셋 클래스
│   │
│   ├──utils/
│    │   ├── __init__.py
│    │  ├── logger.py # 로깅
│    │  ├── metrics.py # 평가 지표
│    │  └── callbacks.py # 콜백 함수
│    └── model.py # 자동화된 모델 학습 및 전처리 .py
│
├── notebooks/
│   └── ~/ # 사용자(~)별 notebook.ipynb가 작성된 곳
│
├── saved/
│   ├── models/ # 저장된 모델
│   ├── predictions/ # 예측 결과
│   └── logs/ # 학습 로그
│
├── app.py # 메인 실행 파일
├── .env.sample # 서버 실행(.sh)을 도와 옵션 설정하는 .env 샘플 파일
├── *.sh # 서버실행을 도와주는 shell script
├── requirements.txt # requirements.txt 라이브러리 설정 파일
├── .gitignore # gitignore file
└── README.md
```

</details>

<details>
    <summary> 라이브러리 버전</summary>

**Python 버전 : 3.10.13**

**Library 버전** - (requirements.txt)

```txt
pandas==2.2.3
python-dateutil==2.9.0.post0
pytz==2024.2
scipy==1.14.1
six==1.16.0
tqdm==4.66.6
typing_extensions==4.12.2
# Recbole
recbole
numpy==1.26.0
torch==2.2.0
ray
kmeans-pytorch
lightgbm
xgboost
omegaconf==2.3.0
# Jupyter notebook
jupyter
# Code Formatting
black
wandb
#pytorch torchvision torchaudio cpuonly -c pytorch
```

</details> 

## 목차
1. 프로젝트 소개
2. EDA
3. Preprocessing
4. Modeling
5. 최종결과
6. 레퍼런스

## 1. 프로젝트 소개

* 프로젝트 기간 : 2024/11/13 ~ 2024/11/28

* 프로젝트 평가 기준 : 10개의 영화 추천에 대한 Recall@K

* 데이터 : MovieLens 데이터를 가공한 대회 제공 데이터(아래 설명 O)

* 프로젝트 개요
> [upstage](https://stages.ai/)의 [Movie Recommendation 대회](https://stages.ai/competitions/339/overview/description) 참가를 위한 프로젝트 <br>
사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화 10개 예측

### 팀 구성 및 역할

> Boostcamp AI Tech 7th RecSys-06 `.fillna` Team

| 이름   | 역할                                         |
|--------|----------------------------------------------|
| [김건율](https://github.com/ChoonB) | 팀장, AutoEncoder 계열 모델링, 앙상블 |
| [백우성](https://github.com/13aek) | ADMMSLIM, LightGCN, S3Rec, BERT4Rec 모델링 및 앙상블          |
| [유대선](https://github.com/xenx96) | Sequential 계열 모델링, 데이터 정제, Github 프로젝트 구조 설계          |
| [이제준](https://github.com/passi3) | Static 계열 모델링, EDA, 데이터 정제, 피처 엔지니어링                               |
| [김성윤](https://github.com/Timeisfast) | EDA & BPR, AutoInt, GRU4RecF 모델링                        |
| [박승우](https://github.com/zip-sa) | EDA & GRU4Rec, Notion 협업 환경 구축, Github 프로젝트 구조 설계                        |

### 프로젝트 진행 과정
1. 프로젝트를 위한 기본 구조 설립 및 코드 작성. (유대선)
2. 각자 데이터 분석 후 토론을 통해 EDA 결과 공유
3. 파트별 역할 분담 진행
4. 추가적인 EDA 진행과 데이터 Preprocessing
5. 모델 실험 및 하이퍼파라미터 튜닝
6. 앙상블
7. 최종 제출 선택

### 협업 방식
- Slack : 팀 간 실시간 커뮤니케이션, [Github 연동] 이슈 공유, 허들에서 실시간 소통 진행
- Zoom : 정기적인 회의와 토론을 위해 사용
- GitHub : 버전 관리와 코드 협업을 위해 사용. 각 팀원은 기능 단위로 이슈를 생성해 이슈 별 브랜치를 만들어 작업하고, Pull Request를 통해 코드 리뷰 후 병합하는 방식으로 진행
- GitHub Projects + Google Calendar : 팀원 간 일정 공유

## 2. EDA
> 팀원 6인 모두 개별적으로 EDA 진행 후 모여서 결과 공유 및 토론

### 2.1 데이터셋 구성 
MovieLens 데이터셋을 기반으로 하며, 총 7개의 주요 파일로 구성

| 파일명 | 행 수 | 주요 내용 |
| --- | --- | --- |
| train_ratings.csv | 5,154,471 | 사용자 시청 이력 (user id, item id, timestamp) |
| titles.tsv | 6,807 | 영화 제목 정보 (item id, title) |
| years.tsv | 6,799 | 영화 개봉 연도 (item id, year) |
| directors.tsv | 5,905 | 영화 감독 정보 (item id, directors) |
| writers.tsv | 11,307 | 영화 작가 정보 (item id, writers) |
| genres.tsv | 15,934 | 영화 장르 정보 (일부 아이템 다중 장르 포함) |
| Ml_item2attributes.json | - | 아이템-속성 매핑 데이터 |

### 2.2 데이터 특징
2.2.1 상호작용 이력 데이터 (train_ratings.csv)
- Implicit feedback 기반 데이터
- 시간 순서가 있는 sequential data
- time은 Unix timestamp(초) 형식

2.2.2 메타데이터
- 영화별 다양한 부가 정보 제공
    - 제목, 개봉년도, 감독, 작가, 장르
- 장르는 다중 레이블 형태 (한 영화가 여러 장르 보유)
- directors와 writers는 고유 ID (nm******) 형식 사용

### 2.3. 테스트 데이터 특성
2.3.1 구성 비율
- Train : Public Test ≈ 100 : 1
- Public Test : Private Test = 1 : 1

2.3.2 주요 특징
- 동일한 사용자 구성 (train/public test/private test)
- 두 가지 예측 대상 포함:
    1. 시간순 다음 아이템 (Next item prediction)
    2. 랜덤 시점 관련 아이템 (Random timestamp item prediction)

### 2.4 데이터 관련 특이사항
2.4.1 기존 MovieLens와 차이
- Explicit feedback(1-5점 평점) 대신 Implicit feedback 사용
- Time-ordered sequence에서 일부 item이 누락
- Side information(메타데이터) 활용 가능
- 추천 문제에서 활용되는 User-Item-Interaction 중 User에 대한 정보 없음

2.4.2 도전 과제
- 사용자의 상호작용 아이템 중 누락된 아이템을 예측하는 Static Model
- 사용자의 마지막 상호작용 아이템을 예측하는 Sequential Model
- 누락된 데이터 처리 전략 수립 필요
- 다양한 메타 데이터의 효과적 활용 방안 모색

> 시간적 순서와 컨텐츠 기반 추천을 모두 고려해야 하는 복합적인 상황

> 데이터 자체가 Explicit feedback이 아닌 Implicit feedback으로, 
아이템에 대한 유저의 단순 상호작용 여부 만을 파악해야하는 문제

### 2.5 기본 EDA
2.5.1 시간에 따른 유저의 아이템 평가

![유저별 상호작용한 아이템의 개수 그래프](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F3f283173-79af-4f57-9401-ee1807e4591b%2Fimage.png?table=block&id=8d5c542c-1404-4a9c-9788-a148355f509c&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

유저별 상호작용한 아이템의 개수 그래프
- 주어진 데이터셋에서 사용자별 상호작용한 아이템의 개수는 멱함수 분포와 유사하게 이루어짐
- 상위 10%의 유저는 335개 이상의 아이템을 평가(상위 20%: 230개, 30%: 176개)

![335개 미만의 아이템을 평가한 유저](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2Faffced3c-899f-4f02-8ad6-a623c2bd2bb6%2Fimage.png?table=block&id=b69efb57-abe3-46e4-91ff-39fccbed5433&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

335개 미만의 아이템을 평가한 유저

- 335개 이하의 아이템을 평가한 유저는 기본적으로 고르게 지수적으로 감소하는 추세를 보이나, 약 45개 미만의 아이템을 평가한 유저는 103명, 전체의 0.3%로 해당 개수 이하 현저히 감소함

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F0fb03298-5ec5-47f1-a625-0e165d03ccab%2Fimage.png?table=block&id=0cae1de4-07bb-4dbd-b2e9-e8803d5facee&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

랜덤한 유저 10명의 일별 리뷰 개수 분포

- 꾸준히 아이템을 평가한 유저도 있지만, 대부분은 단기간 집중적으로 로그가 쌓여있음

2.5.2 월별 아이템 평가 패턴

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F32d1b7ec-6aeb-42a6-b325-a6e8631160ac%2Fimage.png?table=block&id=822daae1-d0b2-4be2-a409-79ece0f5f3f3&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

- 가장 리뷰 수가 많은 Top5 영화에 대해 2008년 10월에 리뷰 수가 급등

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F3a580f2a-8626-43da-9eed-9c8059df984d%2Fimage.png?table=block&id=e3c2895c-f270-4595-b230-09be74e76de9&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

- 리뷰가 수집되기 시작한 2005년 4월 이후에 개봉한 영화에 대해서는 개봉 직후 리뷰 수가 늘었다가 이후 떨어지는 경향이 있음

### 2.6. 심화 EDA 
2.6.1. Genre Co-occurrence Matrix

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F79440e23-13de-4e93-9446-52f6d909affd%2Fimage.png?table=block&id=14c6ac11-5bd2-802f-a9e1-e2133a6ede76&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=660&userId=&cache=v2)

주요 출현 패턴 (Co-occurrence)
- `Drama` 중심 결합: Romance(792), `Comedy`(763), `Thriller`(702)
- `Comedy`-`Romance` 결합: 654회
- `Documentary`: 타 장르와 낮은 결합 빈도

2.6.2. Genre Correlation Matrix

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F4e474f91-7bf4-4af7-b26b-4597cbda77c8%2Fimage.png?table=block&id=14c6ac11-5bd2-80de-8323-fb70428c112e&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=660&userId=&cache=v2)

핵심 상관관계 (Correlation)
- 강한 양의 상관: `Animation`-`Children`(0.84)
- 준-강한 상관: `Crime`-`Thriller`(0.51), `Action`-`Adventure`(0.49)
- 독립적 장르: `Documentary`(대부분 음의 상관)

**요약**
- 범용적 결합성을 가진 장르 (`Drama`)
- 강한 상호 연관성을 가진 장르 쌍 (`Animation`-`Children`, `Crime`-`Thriller`)
- 독립적 특성을 가진 장르 (`Documentary`, `Western`)

2.6.3. Genre Network

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2Ff914026c-7bdf-4ba7-bfbc-eea1ce15f3ed%2FGenre_Network.png?table=block&id=14c6ac11-5bd2-8068-b368-e8fdb17717ca&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=580&userId=&cache=v2)

중심부 클러스터
- **`Drama`-`Comedy`-`Romance`** 삼각형이 네트워크의 핵심을 형성 <br>
→ 이 세 장르가 영화 산업에서 가장 기본적인 장르 조합임을 시사

2.6.4. Genre Popularity Trends

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F59a84244-b984-4cd7-9bf0-3f09ce416721%2FNormalized_Rating_Count.png?table=block&id=14c6ac11-5bd2-8065-aa07-c13ebe211d13&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=660&userId=&cache=v2)

장기 트렌드 관점
- `Drama`의 지속적인 장르 지배력
- `Comedy`의 점진적 인기도 감소
- `Action` 장르의 꾸준한 성장

2.6.5. User Genre Preference Patterns

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2Fe19169b5-c1c6-484f-a2f3-b35e478d1349%2FUser_Genre_Preference_Patterns.png?table=block&id=14c6ac11-5bd2-8033-9fc2-e5fb6aaf8bc5&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

장르 클러스터링 (`Dendrogram`)
- **메인 클러스터**
    - `Drama`, `Romance`가 가장 가까운 거리에서 클러스터링
    - `Adventure`, `Action`이 유사한 패턴을 보임
- **서브 클러스터**
    - `Animation`, `Children`이 하나의 그룹 형성
    - `Crime`, `Thriller`가 유사한 패턴

사용자 선호도 패턴 (`Heatmap`)
- **강한 선호도 패턴**
    - `Drama`에 대한 전반적으로 높은 선호도 (짙은 붉은색)
    - `Action`, `Adventure` 장르의 일관된 선호도
- **희박한 선호도 패턴**
    - `Documentary`, `Film-Noir`, `Western` 등은 매우 낮은 선호도
    - `Musical`, `War` 장르도 상대적으로 낮은 선호도
- **사용자 세그먼트**
    - `Drama` 선호 그룹
    - `Action`/`Adventure` 선호 그룹
    - `Comedy`/`Romance` 선호 그룹의 뚜렷한 구분

2.6.6. Genre Preference Evolution for “User 11”

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F19eebedf-5f6d-4e29-9adf-24417fe39766%2Fimage.png?table=block&id=14c6ac11-5bd2-806d-b77d-c71f606af880&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

**특정 사용자**(`example_user_id` 즉, 본 실험에서는 0번째 행에 위치한 `user 11`):
- 시간대별 장르 선호도 변화
- 각 행은 서로 다른 시간대를 나타냄
- 각 열은 영화 장르를 나타냄
    - 값은 `해당 시간대`에 시청한 `전체 영화` 중 `각 장르가 차지하는 비율(%)`

**전반적인 트렌드:**
- `Action`, `Adventure`는 전 기간에 걸쳐 꾸준히 높은 선호도 유지
- `Sci-Fi`는 `Period 2-3`에서 특히 높은 선호도를 보이다가 변동
- **`Drama`는 시간이 갈수록 증가하는 경향**
- `Documentary`, `Musical`, `Western` 등은 전체적으로 매우 낮은 선호도
- 장르 선호도가 시간에 따라 상당히 동적으로 변화하는 것이 특징

2.6.7 평가 패턴 및 장르 분포 분석

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F26958a0f-071a-46dc-814d-5e9f8efac807%2FSession_Statistics.png?table=block&id=14c6ac11-5bd2-80ea-8bc8-d5d2a231967a&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

### 세션 및 평가 시간 분석 - 상단 그래프

**세션 길이 분포 (Session Length Distribution)와 평가 시간 간격 (Rating Time Intervals)- 상단**

- 대다수의 세션이 50개 이하의 평가로 구성되어 있고, 매우 짧은 세션(10개 미만)이 가장 높은 빈도를 보임 → 한번에 소수의 영화만을 평가하는 경향이 있음을 시사
- **대부분의 평가 간격은 매우 짧은 간격**으로 이루어 짐 → 10^1~2초(약 10~100초)
- 일부 평가는 매우 긴 간격으로 이루어짐 → 10^6초 이상(약 28시간)

→ 이는 사용자들이 '`벌크 평가`'(한 번에 여러 영화 평가)와 '`산발적 평가`' 패턴을 모두 보이고 있음

### 장르 분포와 평가 분석 - 하단 그래프

**벌크 평가의 장르 분포 (Genre Distribution in Bulk Ratings)**

- `Drama`(17.5%), `Comedy`(12%), `Action`(10%) 순으로 높은 비중을 차지
- `Documentary`, `Western`, `Film-Noir`는 가장 낮은 비중(각각 1% 미만)
- 이는 일반적인 장르 선호도가 벌크 평가 행동에도 반영됨

**전체 장르 분포 (Overall Genre Distribution)**

- 벌크 평가의 분포와 유사한 패턴
- `Drama`가 약 22%로 가장 높은 비중을 차지하며, `Comedy`(15%), `Thriller`(9%) 순
- `Western`, `Musical`, `Film-Noir` 등 특수 장르는 전체 평가에서도 낮은 비중을 유지

### 2.8 사용자 데이터 평가 환경 확인

- 사용자와 상호작용된 아이템들 중 일부 아이템이 특정한 시기에 다수의 사용자에게 집중적으로 평가되는 경향이 있음을 확인

> 가정 : 유저가 아이템과 상호작용하는 방식의 차이로 이런 경향이 생기는 것은 아닐까?

![](https://rigorous-shoemaker-76b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Ff94927b0-a808-4d85-ba53-90de2dc55693%2F57fe1a80-20d3-4d3a-baa7-9640ee2b7c53%2Fimage.png?table=block&id=9b73daed-7b1a-4271-825c-bae31de9e39e&spaceId=f94927b0-a808-4d85-ba53-90de2dc55693&width=1420&userId=&cache=v2)

MovieLens의 메인화면

- 데이터가 수집된 MovieLens의 UI는 섹션으로 구분되어 서로 다른 기준으로 다른 아이템들이 추천되고 있음
- 첫 메인 화면에 top picks와 recent release가 구성된다는 점에서 사용자의 평가는 상위 두 개 섹션에 집중되었을 가능성이 높을 것으로 추정됨
- 아이템별 사용자의 최초 리뷰를 기준으로 아이템의 개봉일을 대체

### 2.9 EDA 결론
#### 2.9.1 주요 분석 결과
- **사용자 행동 패턴**
    - 평가 횟수는 멱함수 분포를 따름
    - 상위 10% 사용자가 335개 이상의 영화 평가
    - 대부분의 사용자는 특정 시기에 집중적으로 평가하는 경향
    - 장르 선호도가 시간에 따라 동적으로 변화하는 트렌드를 포착
- **장르 분석**
    - Drama가 가장 범용적인 장르로 확인
    - Animation-Children, Crime-Thriller 등 강한 상관관계 존재
    - Documentary, Western 등은 독립적 특성 보유
- **평가 환경 특성**
    - MovieLens UI의 'top picks'와 'recent release' 섹션이 평가 패턴에 영향
    - 특정 시기에 다수의 사용자가 동일 영화 평가하는 현상 발견

#### 2.9.2 시사점
- **모델링 전략**
    - 시간적 순서와 컨텐츠 기반 추천을 동시 고려 필요
    - **Static/Sequential 모델 각각의 장점 활용 중요**
    - 메타데이터를 효과적으로 활용하는 방안 모색 필요
- **데이터 특성 고려사항**
    - Implicit feedback 특성 반영
    - 누락된 데이터 처리 전략 수립 (directors, writers)
    - 사용자 평가 패턴의 시간적 특성 고려

## 3. Preprocessing
### 3.1 결측치 처리
- 전체 아이템 6807개에 대해 year 데이터에 8개의 결측치 존재
- 결측치에 대해 titles 데이터에서 연도 추출

### 3.2 아이템별 genre 멀티 레이블 인코딩
- genre 데이터에는 모든 아이템에 대한 장르가 최소 1개씩 존재
- 다수의 장르값을 가진 아이템에 대해서는 sklearn의 MultiLabelBinarizer를 이용해 멀티 레이블 인코딩 수행하여 모델 학습

### 3.3 directors, writers 마스킹 파악 및 전처리
- 마스킹된 directors, writers의 값은 동일 인물에 대해 같은 마스킹 값으로 부여되어 있음

```bash
item           0
title          0
director    1304
writer      1159
dtype: int64
```

- 아이템에 따라 director, writer에 결측치 존재 확인
- 해당 결측값 대체 불가능함에 따라 -999로 대체(해당 값은 모델별로 마이너스값 처리 불가능한 경우 0으로 대체)

### 3.4 recent_release 피처 추가

- 유저의 평가 데이터가 수집되는 환경에 맞춰 노출된 영화가 recent release에서 노출된 것인지를 유추하는 피처 생성
- 아이템별 첫 번째 리뷰를 아이템의 개봉일로 가정하고 유저와 아이템의 상호작용 time과 90일 이내일 경우 recent_release를 1로 주는 바이너리 피처

## 4. Modeling
> Team : Static Model / Sequential Model
팀별 역할군 : EDA, Feature Engineering(*after Modeling*), Modeling

- 작업 초기, `BaseLine`과 과제 코드를 기반으로 실험 후 직접 코드를 짜서 모델 구현 시도
- 그러나 직접 구현 시 `Side Information`을 포함한 데이터 셋 정제 문제,  학습 후 추론 관련 차원 문제로 인해 소요되는 시간이 길어짐
- 이에 따라 `RecBole` 라이브러리를 사용해 아래의 모델을 사용하는 방향으로 전환

### 4.1 Static Models
- MultiVAE: Multinomial Cariational Autoencoder for CF with Implicit Feedback
- RecVAE : Variational Autoencoder Model + composite prior distribution + alternating update
- RaCT :  Actor-Critic Pre-training with Autoencoder
- EASE : Embarrassingly Shallow Autoencoders for sparse data (Linear Model)
- LightGCN : simplified GCN Model
- FM : Factorization Machines
- DeepFM : FM + Neural Networks
- BPR : Bayesian Personalized Ranking
- ADMMSLIM : efficient Optimization approach for Sparse Linear Methods using ADMM
- AutoInt : Attention-based model that automatically learns high-order feature interactions

### 4.2 Sequential Models
- SASRec : Self-Attention-Sequential Recommendation
- SASRecF : SASRec + Side Information
- S3Rec : Self-Supervised Learning for Sequential Recommendation
- BERT4Rec : Bidirectional Encoder Representations from Transformer
- GRU4RecF : Session based recommendations with recurrent neural network

| **모델명** | valid score | public score | 특이사항 |
| --- | --- | --- | --- |
| RecVae | x | 0.1380 | hidden_dimension: 600 / latent_dimension: 200 |
| RaCT | 0.1433 | 0.1265 | side information 반영 |
| MultiVAE | 0.1443 | 0.1278 | learning_rate: 0.001, train_batch_size: 256,embedding_size: 64 |
| GRU4Rec | 0.0776 | x | embedding & hidden_size: 64, num_layers: 1, dropout_prob: 0.1 |
| BERT4Rec | 0.1299 | 0.0937 | RecBole 기본 세팅 |
| LightGCN | 0.1383 | x | train-valid-test (1.0:0.0:0.0)  |
| DeepFM | x | 0.0310 | 상호작용된 아이템이 추천되어 보수 작업 진행 중 제외 |
| FM | 0.1269 | 0.0075 | 상호작용된 아이템이 추천되어 보수 작업 진행 중 제외 |
| SASRecF | 0.058 | x | Genres One-HotCoding & Neg-Sample dist(popularity) |
| BPR | 0.1229 | 0.1068 | RecBole 기본 세팅 |
| ADMMSLIM | x | 0.1577 | train-valid-test (1.0:0.0:0.0) |
| EASE | x | 0.1594 | train-valid-test (1.0:0.0:0.0) |
| GRU4RecF | 0.1666 | 0.1011 | embedding_size: 64, hidden_size: 128, interaction + genre, title |
| AutoInt | 0.1246 | x | interaction + genre, title, director, writer |
| SASRecF | 0.0952 | x | Genres One-HotCoding, negative-sampling-number : 5 |
| S3Rec | x | 0.0876 | title, year, genre, director, writer |

### 4.3 Ensemble
> EASE(**E**), RecVAE(**R**), BERT4Rec(**B**), LightGCN(**L**), ADMMSLIM(**S**), Autoint(**A**), BPR(**P**), GRU4Rec(**G**)<br>
실험 했던 모델 중 앙상블은 위의 모델 사용. (괄호 안은 약자)<br>
각 모델에서 유저 한 명 당 영화 Top 20을 뽑아서 Voting하는 방식으로 앙상블 진행

- Hard Voting은 1위는 1점, 20위는 0.05점 식으로 순위 별 차등 포인트를 줘서 합산하는 방식

- Soft Voting은 각 모델에서 랭킹을 뽑는 데 사용한 score를 표준화해서 합산하는 방식
- 앙상블 실험 결과(주요 실험 일부만 정리)

| 사용한 모델 | 방식 | public score | private score |
| --- | --- | --- | --- |
| ERBLSAP | Hard Voting | 0.1585 | 0.1567 |
| ES | Soft Voting | 0.1584 | 0.1578 |
| ESG | Soft Voting | 0.1647 | 0.1544 |
| ESGB | Soft Voting | 0.1628 | 0.1509 |
| ESG | Hard Voting | 0.1698 | 0.1654 |
| ESGR | Hard Voting | 0.1710 | 0.1679 |
| ESGRABPL | Hard Voting(A4BL은 포인트 50% 적용) | 0.1649 | 0.1623 |
| ESGR | Hard Voting(1~10위는 1점, 11~20위는 0.5점) | 0.1675 | 0.1654 |
| ESGR3 | Hard Voting | 0.1703 | 0.1671 |

## 5. 최종 결과
- Public Score가 가장 높았던 EASE + ADMMSLIM + GRU4Rec + RecVAE 모델을 하드 보팅한 방식과 S3Rec까지 포함한 방식 2가지를 제출
- ESGR Hard Voting : Public Score Recall@10 =  0.1710 / Private Score = 0.1679
- ESGR3 Hard Voting : Public Score Recall@10 =  0.1703 / Private Score = 0.1671
- **Private Score 1위 / Public Score 2위**
- Sequential 계열 모델의 경우 개별 public 스코어는 낮았으나, 점수가 높았던 AutoEncoder 모델의 결과와 앙상블한 결과의 점수가 가장 높음
    - 주어진 문제 자체가 static과 sequential의 두 가지 접근을 요하는 문제이다보니, 정답에서 상대적으로  개수가 적은 sequential 모델의 독자적인 성능이 떨어질 수 없었을 것이라 추정

## 6. 레퍼런스
- RecBole Library : https://github.com/RUCAIBox/RecBole
- MultiVAE :  https://github.com/younggyoseo/vae-cf-pytorch
