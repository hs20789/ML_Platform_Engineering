# %%
import pandas as pd
import mlflow
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

# %%
try:
    CURRENT_DIR = Path(__file__).resolve().parent
except NameError:
    CURRENT_DIR = Path.cwd()

DATA_DIR = CURRENT_DIR.parent / "data"
PLOT_DIR = CURRENT_DIR / "categorical_variable_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(DATA_DIR / "income_data.csv", skipinitialspace=True)
print(df.head(), '\n')
print(df.info(), '\n')


# MLflow 추적
# %%
import uuid

# %%
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow 서버 URI 설정
mlflow.set_experiment("Income Prediction Experiment")  # 실험 이름 설정

# %%
with mlflow.start_run(run_name=f"eda-{uuid.uuid4()}"):

    for column in df.drop(columns=["Target"]).select_dtypes(include="object").columns:

        print(f"Variable {column}\n")
        print(df[column].value_counts())

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            data=df,
            y=column,
            hue="Target",
            multiple="fill",
            ax=ax
        )

        ax.set_title(f"Variable {column} ~ Target")
        ax.set_xlabel("Proportion")

        fig.tight_layout()

        fig.savefig(PLOT_DIR / f"Variable {column}.png")

        plt.show()
        plt.close(fig)

    mlflow.log_artifacts(PLOT_DIR)
    
    
# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

target = df.Target
feature_df = df.drop('Target', axis=1)

encoder = OneHotEncoder(sparse_output=False, drop='if_binary')
target = encoder.fit_transform(np.array(target).reshape(-1, 1))
dummyfied_df = pd.get_dummies(feature_df, drop_first=True, sparse=False, dtype=float)
coll_list = dummyfied_df.columns.to_list()
with open('column_list.pkl', 'wb') as f:
    pickle.dump(coll_list, f)
X_train, X_test, y_train, y_test = train_test_split(dummyfied_df.reindex(columns=coll_list, fill_value=0), target, test_size=0.2, shuffle=True, random_state=42)

# %%
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# NCP Object Storage Client
s3_client = boto3.client(
    "s3",
    endpoint_url="https://kr.object.ncloudstorage.com",
    region_name="kr-standard",
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
bucket_name = "mlflow-datasets-heonsu"

try:
    s3_client.head_bucket(Bucket=bucket_name)
except ClientError as e:
    # HEAD 응답은 본문이 없어서 에러 코드가 HTTP 상태코드 문자열("404", "403")로 온다
    if e.response["Error"]["Code"] == "404":
        print("PLEASE CREATE BUCKET IN NCP OBJECT STORAGE BEFORE PROCEEDING")
    else:
        raise  # 403(키/권한 문제) 등은 그대로 드러낸다


# DataFrame을 CSV로 변환해서 NCP Object Storage에 저장
def save_df_to_ncp(df, bucket_name, path):
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    s3_client.put_object(
        Bucket=bucket_name,
        Key=path,
        Body=csv_bytes,
        ContentType="text/csv",
    )
