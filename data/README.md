# Dataset

The raw dataset is **not included** in this repository. Download it from Kaggle and place it here.

| | |
|---|---|
| **Name** | Uber Fares Dataset |
| **Publisher** | Yasser H. (Kaggle user `yasserh`) |
| **Source** | https://www.kaggle.com/datasets/yasserh/uber-fares-dataset |
| **License (as listed on Kaggle)** | CC0: Public Domain |
| **File** | `uber.csv`, about 23 MB, 200,000 rows × 9 columns |
| **Content** | Uber rides associated with New York City, 2009–2015 |

## Setup

1. Download the dataset from the Kaggle page above. A free Kaggle account is required. Alternatively, use the
   Kaggle CLI: `kaggle datasets download -d yasserh/uber-fares-dataset --unzip`
2. Put the file at:

   ```
   uber-fare-prediction-mlp/data/uber.csv
   ```

3. Run `Uber_Fare_Prediction.ipynb` from the repository root.

## Columns

| Column | Description |
|---|---|
| `Unnamed: 0`, `key` | record index and key (not used by the model) |
| `fare_amount` | ride fare in USD (target) |
| `pickup_datetime` | pickup date and time |
| `pickup_longitude`, `pickup_latitude` | pickup coordinates |
| `dropoff_longitude`, `dropoff_latitude` | drop-off coordinates |
| `passenger_count` | number of passengers |

The file must keep the original name and column names. `data/*.csv` is excluded from Git by `.gitignore`.

The dataset was published on Kaggle by its uploader. It was not collected or created by the author of this
repository.
