# %%
import pandas as pd
import mlflow
from pathlib import Path
import os

# %%
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "income_data.csv")
print(df.head(), '\n')
print(df.info(), '\n')

os.makedirs("categorical_variable_plots", exist_ok=True)
for i in df.iloc[:, :-1].select_dtypes(include='object').columns:
    print(f"Variable {i} \n")
    print(df[i].value_contents)