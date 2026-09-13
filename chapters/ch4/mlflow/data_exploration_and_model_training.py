# %%
import pandas as pd
import mlflow
from pathlib import Path
import os
import time  # Random Forest 학습 소요 시간(fit_time_seconds) 측정용
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# 터미널 cwd든 VS Code 인터랙티브 실행이든 결과가 같도록,
# pyproject.toml과 mlflow 폴더가 함께 있는 "프로젝트 루트"를 찾아 올라간다.
try:
    start_dir = Path(__file__).resolve().parent
except NameError:
    start_dir = Path.cwd().resolve()

for folder in (start_dir, *start_dir.parents):
    if (folder / "pyproject.toml").is_file() and (folder / "mlflow").is_dir():
        PROJECT_DIR = folder
        break
else:
    raise FileNotFoundError("프로젝트 폴더 안에서 실행해 주세요.")

CURRENT_DIR = PROJECT_DIR / "mlflow"  # 이 챕터 코드/산출물 폴더
DATA_DIR = PROJECT_DIR / "data"  # 원본 CSV 위치
PLOT_DIR = CURRENT_DIR / "categorical_variable_plots"  # EDA 플롯 png 저장 위치 (뒤에서 MLflow 아티팩트로 업로드)
os.makedirs(PLOT_DIR, exist_ok=True)

# income_data.csv: Census Income(Adult) 데이터셋. Target(<=50K/>50K)을 맞히는 이진분류 문제
df = pd.read_csv(DATA_DIR / "income_data.csv", skipinitialspace=True)
print(df.head(), "\n")  # 컬럼 구성 육안 확인
print(df.info(), "\n")  # 결측치·dtype 확인


# MLflow 추적
# 이 파일은 "Income Prediction Experiment" 실험 하나 안에 EDA 1개 + 모델 학습 3개,
# 총 4개의 run을 기록한다. (run = 실험을 한 번 시도한 기록 단위, 서로 지표로 비교 가능)
# %%
import uuid

# %%
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow 서버 URI 설정 (로컬에 mlflow server가 떠 있어야 함)
mlflow.set_experiment("Income Prediction Experiment")  # 실험 이름 설정. 파일 끝 MlflowClient 조회부와 이름이 반드시 같아야 함

# %%
# --- Run 1/4: EDA ---
# 범주형 변수별로 Target 비율이 어떻게 갈리는지 그림으로 남겨서,
# 어떤 변수가 예측에 쓸모 있어 보이는지 MLflow UI에서 사람이 확인할 수 있게 한다.
with mlflow.start_run(run_name=f"eda-{uuid.uuid4()}"):
    for column in df.drop(columns=["Target"]).select_dtypes(include="object").columns:
        print(f"Variable {column}\n")
        print(df[column].value_counts())

        fig, ax = plt.subplots(figsize=(8, 5))

        # multiple="fill": 막대를 100% 기준으로 채워서 카테고리별 Target 비율을 바로 비교할 수 있게 함
        sns.histplot(data=df, y=column, hue="Target", multiple="fill", ax=ax)

        ax.set_title(f"Variable {column} ~ Target")
        ax.set_xlabel("Proportion")

        fig.tight_layout()

        fig.savefig(PLOT_DIR / f"Variable {column}.png")

        plt.show()
        plt.close(fig)

    # 로컬에 쌓아둔 png들을 이 run의 아티팩트로 한 번에 업로드
    mlflow.log_artifacts(PLOT_DIR)


# %%
# --- 전처리: 뒤에 나올 DT/RF/XGBoost 세 모델이 공통으로 쓸 학습 데이터를 한 번만 준비 ---
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

target = df.Target
feature_df = df.drop("Target", axis=1)

# Target(<=50K/>50K, 이진값)을 0/1 컬럼 하나로 인코딩
encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
target = encoder.fit_transform(np.array(target).reshape(-1, 1))
# 나머지 범주형 피처는 원-핫(더미) 인코딩
dummyfied_df = pd.get_dummies(feature_df, drop_first=True, sparse=False, dtype=float)
coll_list = dummyfied_df.columns.to_list()

# 학습에 쓴 컬럼 구성을 저장해둔다.
# 나중에 이 모델로 추론할 새 데이터를 같은 컬럼 순서·구성으로 맞추기(reindex) 위한 기준표 역할.
COLUMN_LIST_PATH = CURRENT_DIR / "column_list.pkl"
with open(COLUMN_LIST_PATH, "wb") as f:
    pickle.dump(coll_list, f)

X_train, X_test, y_train, y_test = train_test_split(
    dummyfied_df.reindex(columns=coll_list, fill_value=0),
    target,
    test_size=0.2,
    shuffle=True,
    random_state=42,  # 실행할 때마다 같은 방식으로 분할되도록 시드 고정
)

# %%
# --- NCP Object Storage 연동 준비 ---
# 원본 예제는 MinIO(자체 호스팅 S3 호환 스토리지) 기준이었는데,
# 여기서는 네이버클라우드 Object Storage(S3 호환 API)를 그대로 쓰도록 바꿨다.
# boto3 "s3" 클라이언트의 endpoint_url만 NCP 주소로 지정하면 이후 API 호출은 동일하다.
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# NCP Object Storage Client
s3_client = boto3.client(
    "s3",
    endpoint_url="https://kr.object.ncloudstorage.com",
    region_name="kr-standard",
    # 실행 전 터미널에서 export NCP_ACCESS_KEY_ID=... / NCP_SECRET_ACCESS_KEY=... 로 지정 필요
    aws_access_key_id=os.environ["NCP_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["NCP_SECRET_ACCESS_KEY"],
    # boto3 1.36+는 업로드 시 기본으로 체크섬 트레일러(aws-chunked)를 붙이는데,
    # NCP는 이를 AccessDenied로 거부한다. 필요할 때만 체크섬을 쓰도록 되돌린다.
    config=Config(
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)


# Bucket 존재 확인
bucket_name = "mlflow-datasets-heonsu"  # 아래 세 모델 학습 run에서 BUCKET_NAME으로 재사용됨

try:
    s3_client.head_bucket(Bucket=bucket_name)
except ClientError as e:
    # HEAD 응답은 본문이 없어서 에러 코드가 HTTP 상태코드 문자열("404", "403")로 온다
    if e.response["Error"]["Code"] == "404":
        print("PLEASE CREATE BUCKET IN NCP OBJECT STORAGE BEFORE PROCEEDING")
    else:
        raise  # 403(키/권한 문제) 등은 그대로 드러낸다


# DataFrame을 CSV로 변환해서 NCP Object Storage에 저장
# (로컬에 임시 파일을 만들지 않고, 메모리 안에서 바로 CSV 바이트로 인코딩해서 업로드한다)
def save_df_to_ncp(df, bucket_name, path):
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    s3_client.put_object(
        Bucket=bucket_name,
        Key=path,
        Body=csv_bytes,
        ContentType="text/csv",
    )


# %%
# --- Run 2/4: Decision Tree ---
# 이 run에서 하는 일: (1) 이번 run이 쓰는 데이터를 NCP에 올리고 MLflow에 lineage로 기록
#                    (2) 모델 학습
#                    (3) 성능 지표 계산 후 MLflow에 기록
#                    (4) 모델 자체와 하이퍼파라미터를 MLflow에 저장
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

BUCKET_NAME = bucket_name  # 위에서 존재를 확인한 버킷을 그대로 재사용
with mlflow.start_run() as run:
    tree = DecisionTreeClassifier()
    run_id = run.info.run_id  # NCP 업로드 파일명에 넣어서 run끼리 겹치지 않게 함

    # 1) 이번 run이 사용하는 원본/학습/평가 데이터를 NCP Object Storage에 CSV로 업로드
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    save_df_to_ncp(feature_df, BUCKET_NAME, feature_df_path)
    train_df = pd.concat([X_train, pd.Series(y_train.ravel())], axis=1)
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_ncp(train_df, BUCKET_NAME, train_df_path)  # A
    test_df = pd.concat([X_test, pd.Series(y_test.ravel())], axis=1)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_ncp(test_df, BUCKET_NAME, test_df_path)

    # 2) 방금 NCP에 올린 위치(source)를 MLflow Dataset 객체로 감싸서
    #    이 run이 정확히 어떤 데이터로 학습·평가됐는지 계보(lineage)를 남긴다
    training_dataset = mlflow.data.from_pandas(
        train_df, source=f"{BUCKET_NAME}/{train_df_path}"
    )  # B
    test_dataset = mlflow.data.from_pandas(
        test_df, source=f"{BUCKET_NAME}/{test_df_path}"
    )
    feature_dataset = mlflow.data.from_pandas(
        feature_df, source=f"{BUCKET_NAME}/{feature_df_path}"
    )
    mlflow.log_input(training_dataset, context="training")  # C
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_input(feature_dataset, context="reference")

    # 3) 실제 모델 학습
    tree.fit(X_train, y_train.ravel())  # D

    # 4) 성능 평가: ROC-AUC(확률 기반)와 정확도를 train/test 둘 다 계산
    #    (train·test 차이가 크면 과적합 의심)
    roc_auc_score_train = roc_auc_score(
        y_train == 1, tree.predict_proba(X_train)[:, 1]
    )  # E
    roc_auc_score_test = roc_auc_score(y_test == 1, tree.predict_proba(X_test)[:, 1])
    training_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)

    # 5) 학습에 쓴 컬럼 목록도 아티팩트로 남겨서, 이 모델로 추론할 때 입력 컬럼을 맞출 수 있게 함
    mlflow.log_artifact(COLUMN_LIST_PATH)

    # 6) 지표 기록 — MLflow UI에서 run끼리 이 숫자들로 비교하게 됨
    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)  # F
    print(f"Roc Auc Score train: {roc_auc_score_train}  \n")
    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}  \n")
    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train : {training_accuracy}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test : {test_accuracy}")

    # 7) 학습된 모델과 하이퍼파라미터 저장.
    #    artifact_path="income-classifier" → 파일 끝 model_artifact_path와 이름이 같아야 register_model이 찾을 수 있음
    mlflow.sklearn.log_model(tree, "income-classifier")  # G
    mlflow.log_params(tree.get_params())  # H


# %%
# --- Run 3/4: Random Forest ---
# 데이터 업로드 → lineage 기록 → 학습 → 평가 → 저장까지 Decision Tree와 완전히 같은 패턴.
# 차이점: 트리를 여러 개 앙상블하는 만큼 학습 시간이 궁금해질 수 있어 fit 소요 시간도 같이 기록한다.
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run() as run:
    forest = RandomForestClassifier()

    start = time.time()
    run_id = run.info.run_id
    train_df = pd.concat([X_train, pd.Series(y_train.ravel())], axis=1)
    test_df = pd.concat([X_test, pd.Series(y_test.ravel())], axis=1)
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_ncp(train_df, BUCKET_NAME, train_df_path)
    save_df_to_ncp(feature_df, BUCKET_NAME, feature_df_path)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_ncp(test_df, BUCKET_NAME, test_df_path)
    feature_dataset = mlflow.data.from_pandas(
        feature_df, source=f"{BUCKET_NAME}/{feature_df_path}"
    )
    training_dataset = mlflow.data.from_pandas(
        train_df, source=f"{BUCKET_NAME}/{train_df_path}"
    )
    test_dataset = mlflow.data.from_pandas(
        test_df, source=f"{BUCKET_NAME}/{test_df_path}"
    )
    mlflow.log_input(training_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")
    mlflow.log_input(feature_dataset, context="reference_features")

    forest.fit(X_train, y_train.ravel())
    end = time.time()
    mlflow.log_metric("fit_time_seconds", end - start)  # Decision Tree 대비 학습 시간 비교용

    roc_auc_score_train = roc_auc_score(
        y_train == 1, forest.predict_proba(X_train)[:, 1]
    )
    roc_auc_score_test = roc_auc_score(y_test == 1, forest.predict_proba(X_test)[:, 1])
    training_accuracy = forest.score(X_train, y_train)
    test_accuracy = forest.score(X_test, y_test)
    mlflow.log_artifact(COLUMN_LIST_PATH)
    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)
    print(f"Roc Auc Score train: {roc_auc_score_train}  \n")
    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}  \n")
    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train : {training_accuracy}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test : {test_accuracy}")
    # artifact_path="income-classifier" → Decision Tree와 동일한 이름으로 저장 (파일 끝 등록 단계와 연결됨)
    mlflow.sklearn.log_model(forest, "income-classifier")

    mlflow.log_params(forest.get_params())


# %%
# --- Run 4/4: XGBoost (autolog 사용) ---
# 앞의 두 모델은 log_input/log_model/log_params를 전부 손으로 호출했지만,
# 여기서는 mlflow.xgboost.autolog() 한 줄로 파라미터·모델 등을 자동 기록되게 한다.
# 다만 ROC-AUC처럼 우리가 직접 정의한 지표는 autolog가 알 수 없어서 여전히 수동으로 log_metric 해야 한다.
import xgboost as xgb
from sklearn.metrics import accuracy_score

with mlflow.start_run() as run:
    mlflow.xgboost.autolog()  # A  # DMatrix 생성보다 먼저 호출해야 자동 기록이 제대로 걸림
    n_round = 30
    run_id = run.info.run_id
    dtrain = xgb.DMatrix(data=X_train, label=y_train.ravel())
    dtest = xgb.DMatrix(data=X_test, label=y_test.ravel())

    # 1차 학습: binary:logistic → 확률값(0~1)을 출력해서 ROC-AUC 계산에 적합
    params = {
        "objective": "binary:logistic",
        "colsample_bytree": 1,
        "learning_rate": 1,
        "max_depth": 10,
        "subsample": 1,
    }
    model = xgb.train(params, dtrain, n_round)
    ax = xgb.plot_importance(model, max_num_features=10, importance_type="cover")
    fig = ax.figure
    fig.set_size_inches(10, 8)
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)

    # 2차 학습: binary:hinge로 다시 학습해서 위 model/pred_train/pred_test를 덮어쓴다.
    # 주의: hinge는 0/1 하드 라벨을 출력하므로, 아래 roc_auc_score는 방금 위에서 구한
    #       1차(logistic, 확률 기반) 예측이 아니라 이 2차 결과로 계산된다.
    #       (1차 학습·예측은 위 feature importance 플롯에만 쓰이고 최종 지표엔 반영되지 않음 — 원본 코드의 특징)
    model = xgb.train(
        params={
            "objective": "binary:hinge",
            "colsample_bytree": 1,
            "learning_rate": 1,
            "max_depth": 10,
            "subsample": 1,
        },
        dtrain=dtrain,
    )
    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)
    roc_auc_score_train = roc_auc_score(y_train == 1, pred_train)
    roc_auc_score_test = roc_auc_score(y_test == 1, pred_test)
    training_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)
    mlflow.log_metric("roc_auc_score_train", roc_auc_score_train)  # B
    mlflow.log_artifact(COLUMN_LIST_PATH)
    print(f"Roc Auc Score train: {roc_auc_score_train}  \n")
    mlflow.log_metric("roc_auc_score_test", roc_auc_score_test)
    print(f"Roc Auc Score test: {roc_auc_score_test}  \n")
    mlflow.log_metric("training_accuracy", training_accuracy)
    print(f"Accuracy train : {training_accuracy}")
    mlflow.log_metric("test_accuracy", test_accuracy)
    print(f"Accuracy test : {test_accuracy}")

# %%
# --- 마무리: 4개 run 중 가장 성능 좋은 모델을 Model Registry에 등록 ---
# 지금까지는 "학습하고 기록"만 했지, 실제로 서빙에 쓸 모델을 고르진 않았다.
# 여기서 실험 안의 모든 run을 검색해 test ROC-AUC가 0.8을 넘는 것 중 1등을 찾고,
# 그 run이 저장한 모델을 "random-forest-classifier"라는 이름으로 정식 등록(버전 1, 2, ... 부여)한다.
from mlflow import MlflowClient

mlflow_client = MlflowClient()
experiment_name = "Income Prediction Experiment"  # 38번 줄 set_experiment와 동일해야 조회됨
model_artifact_path = "income-classifier"  # mlflow.sklearn.log_model(..., "income-classifier")에서 쓴 이름
experiment = mlflow_client.get_experiment_by_name(experiment_name)
run_object = mlflow_client.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string="metrics.roc_auc_score_test > 0.8",  # 조건을 만족하는 run이 하나도 없으면 아래 [0]에서 IndexError
    max_results=1,
    order_by=["metrics.roc_auc_score_test DESC"],  # 내림차순 정렬해서 1등만 가져옴
)[0]
# 주의: XGBoost run이 1등이 되면 이 경로에 모델이 없어 register_model이 실패할 수 있다
# (XGBoost 블록은 autolog에만 의존하고 "income-classifier"라는 이름으로 명시적으로 log_model하지 않기 때문)
model_uri = f"runs:/{run_object.info.run_id}/{model_artifact_path}"
mlflow.register_model(model_uri, "random-forest-classifier")
