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

df = pd.read_csv(DATA_DIR / "income_data.csv", skipinitialspace=True)
print(df.head(), '\n')
print(df.info(), '\n')

os.makedirs(CURRENT_DIR / "categorical_variable_plots", exist_ok=True)
for i in df.drop(columns=["Target"]).select_dtypes(include='object').columns:
    print(f"Variable {i} \n")
    print(df[i].value_counts(), '\n')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, y=i, hue="Target", multiple="fill", ax=ax)
    ax.set_title(f"Distribution of {i} by Target Variable")
    ax.set_xlabel("Proportion")
    fig.savefig(CURRENT_DIR / "categorical_variable_plots" / f"{i}_distribution.png")
    plt.close(fig)


# MLflow 추적
# %%
import mlflow
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

        fig.savefig(
            CURRENT_DIR / "categorical_variable_plots" / f"Variable {column}.png"
        )

        plt.show()
        plt.close(fig)

    mlflow.log_artifacts(CURRENT_DIR / "categorical_variable_plots")