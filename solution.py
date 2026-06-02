"""
Reproduce the 0.95458 submission for Playground Series S6E5 (F1 pit stops).

Idea: the best public submission caps at 0.95454 and blending more public files
only drags it down. A model trained on the original real F1 dataset is barely
correlated with those public blends but still carries clean signal, so a small
slice of it corrects the anchor's ranking mistakes.

    final = 0.95 * rank(public_best) + 0.05 * rank(orig_model)

Inputs (pulled with the kaggle CLI, needs ~/.kaggle/credentials.json):
  - competition test set: playground-series-s6e5
  - public anchor:        raunakdey07/f1-pit-stops-0-95454
  - original dataset:     aadigupta1601/f1-strategy-dataset-pit-stop-prediction

Run:  python solution.py   ->  writes O2.csv
"""
import os
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "inputs")
os.makedirs(DATA, exist_ok=True)


def kaggle(args):
    subprocess.run(["kaggle"] + args, check=True)


def fetch():
    if not os.path.exists(os.path.join(DATA, "test.csv")):
        kaggle(["competitions", "download", "-c", "playground-series-s6e5",
                "-p", DATA])
        subprocess.run(["unzip", "-o",
                        os.path.join(DATA, "playground-series-s6e5.zip"),
                        "-d", DATA], check=True)
    if not os.path.exists(os.path.join(DATA, "raunak", "submission.csv")):
        kaggle(["datasets", "download", "-d", "raunakdey07/f1-pit-stops-0-95454",
                "-p", os.path.join(DATA, "raunak"), "--unzip"])
    if not os.path.exists(os.path.join(DATA, "orig", "f1_strategy_dataset_v4.csv")):
        kaggle(["datasets", "download", "-d",
                "aadigupta1601/f1-strategy-dataset-pit-stop-prediction",
                "-p", os.path.join(DATA, "orig"), "--unzip"])


NUM = ["LapNumber", "Stint", "TyreLife", "Position", "LapTime",
       "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
       "Position_Change", "PitStop"]
CAT = ["Driver", "Compound", "Race", "Year_c"]


def feats(df):
    df = df.rename(columns={"LapTime (s)": "LapTime"}).copy()
    for c in ["Driver", "Compound", "Race"]:
        df[c] = df[c].astype("category").cat.codes
    df["Year_c"] = df["Year"]
    return df[NUM + CAT]


def rank(a):
    return rankdata(a) / len(a)


def main():
    fetch()
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    ids = test["id"].values

    # strongest public submission (0.95454)
    raunak = (pd.read_csv(os.path.join(DATA, "raunak", "submission.csv"))
              .set_index("id")["PitNextLap"].reindex(ids).values)

    # model on the original clean F1 data
    orig = pd.read_csv(os.path.join(DATA, "orig", "f1_strategy_dataset_v4.csv"))
    params = dict(objective="binary", metric="auc", learning_rate=0.03,
                  num_leaves=64, min_data_in_leaf=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, n_jobs=8,
                  verbose=-1, seed=42)
    model = lgb.train(params,
                      lgb.Dataset(feats(orig), orig["PitNextLap"].astype(int).values),
                      num_boost_round=600)
    orig_pred = model.predict(feats(test))

    print("rank correlation public vs original model:",
          round(np.corrcoef(rank(raunak), rank(orig_pred))[0, 1], 4))

    blend = 0.95 * rank(raunak) + 0.05 * rank(orig_pred)
    out = os.path.join(HERE, "O2.csv")
    pd.DataFrame({"id": ids.astype(int), "PitNextLap": blend}).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
