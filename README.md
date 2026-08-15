# Chicago Crime Analysis and Arrest Prediction

An ML1 (introductory Machine Learning) college project that explores the
Chicago Crime dataset and builds a simple classifier to predict whether a
reported crime results in an arrest.

## Objective

The project has two parts:

1. **Exploratory Data Analysis (EDA)** — understand the dataset: data quality,
   crime types, arrest rates, and how crime patterns change over time and
   location.
2. **Machine Learning** — train a binary classifier that predicts
   `Arrest` (`1 = Arrest`, `0 = No Arrest`) for a reported crime, using only
   information that would be known **at the time the crime is reported** (no
   information from after the case was processed).

## Dataset

`data/chicago_crime.csv` — Chicago Police Department crime records (2021–2025
in this extract), one row per reported incident, with columns such as `Date`,
`Primary Type`, `Description`, `Location Description`, `District`, `Ward`,
`Community Area`, `Beat`, `Latitude`/`Longitude`, and the target column
`Arrest`.

## Technologies

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Streamlit
- Plotly (interactive charts)

## EDA

The app inspects the raw data (shape, dtypes, missing values, duplicates),
cleans it conservatively (fills missing values, blanks impossible
coordinates, never drops large chunks of data without reason), and then
analyzes it:

- **Univariate**: crime type frequency, arrest distribution, descriptions.
- **Temporal**: crimes by year, month, hour, day of week.
- **Location**: top districts, top location types, arrest rate by district,
  a sampled scatter map.
- **Bivariate**: arrest rate by crime type / district / hour / day / month.
- **Multivariate**: hour-by-day heatmap, crime-type-by-district arrest-rate
  heatmap, crime-type-by-hour heatmap.
- **Automatic insights**: a list of findings computed live from the data
  (never hard-coded).
- **3D Crime Intelligence Map** (signature feature, page 10): Chicago is cut
  into a lat/lon grid (Low/Medium/High resolution), crimes are aggregated per
  cell, and each cell becomes a 3D "tower" in a Plotly `Mesh3d` skyline —
  taller towers mean more reported crime. Color can be switched between
  Crime Volume, Arrest Rate and Crime Type. It includes its own filters
  (Year, Crime Type, District, Arrest Status, Grid Resolution), a 2D/3D view
  toggle, hover details per tower, a dropdown-based hotspot analysis panel
  (3D click-selection is unreliable in Streamlit, so a dropdown is used as a
  reliable fallback), a two-district 3D comparison, and an optional
  year/month "crime over time" view. All aggregation is cached with
  `st.cache_data` so the map stays responsive on the full dataset.

## ML

- **Target variable**: `Arrest` → `ArrestFlag` (1 = Arrest, 0 = No Arrest).
- **Feature engineering**: `Year`, `Month`, `Hour`, `DayOfWeek` extracted from
  `Date`; categorical features `Primary Type`, `Description`,
  `Location Description`, `District`, `Ward`, `Community Area`, `Beat`,
  `Domestic`.
- **Excluded columns** (and why): `ID`/`Case Number` (identifiers),
  `Updated On` (recorded after the case is processed — leakage), `Date`
  (replaced by extracted features), `Block`/`Location`/`X,Y Coordinate`
  (redundant with District/Beat or Latitude/Longitude), `Latitude`/`Longitude`
  (District/Beat/Ward already describe location), `IUCR`/`FBI Code`
  (duplicate offence codes), `_year` (duplicate of `Year`). See the
  "Machine Learning" page in the app for the full list and reasons.
- **Preprocessing**: `ColumnTransformer` with
  `SimpleImputer(median) → StandardScaler` for numerical features and
  `SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")` for
  categorical features, wrapped in a single `Pipeline` with the classifier.
- **Train/test split**: 80/20, `stratify=y`, `random_state=42`.
- **Models**: Logistic Regression (`max_iter=1000`) and Decision Tree
  (`max_depth=10`).
- **Validation**: 5-fold `StratifiedKFold` cross-validation on the training
  set (accuracy, precision, recall, F1 — mean ± std).
- **Test evaluation**: accuracy, precision, recall, F1, confusion matrix, and
  ROC-AUC as a secondary metric.
- **Best model selection**: automatic, based on the highest **test F1-score**
  (accuracy alone is misleading here because roughly two-thirds of crimes do
  not end in an arrest).

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the dataset from `data/chicago_crime.csv`. A sidebar option
lets you switch from the full dataset to a smaller random sample
(`random_state=42`) if your machine is slow.

## Project Structure

```text
chicago-crime-ml/
│
├── app.py                  # Streamlit app (UI + page layout only)
├── requirements.txt
├── README.md
│
├── data/
│   └── chicago_crime.csv   # dataset
│
├── src/
│   ├── data_loader.py      # CSV loading, overview/missing/duplicate reports
│   ├── preprocessing.py    # cleaning, feature engineering, sklearn pipeline
│   ├── eda.py               # all Plotly charts + automatic insight generator
│   ├── model.py             # train/test split, CV, evaluation, prediction
│   └── map3d.py             # grid aggregation + 3D Crime Skyline (Mesh3d)
│
└── outputs/                 # (reserved for any exported figures/results)
```

## Results

Results below are illustrative — the app recalculates and displays the exact
numbers live in the **Machine Learning** and **Model Comparison** pages
whenever it runs (they change slightly with sampling and dataset updates).

| Model               | Test Accuracy | Test Precision | Test Recall | Test F1 |
| ------------------- | -------------:| ---------------:| -----------:| -------:|
| Logistic Regression |        ~0.82 |            ~0.83 |       ~0.60 |   ~0.70 |
| Decision Tree       |        ~0.81 |            ~0.84 |       ~0.52 |   ~0.64 |

**Best model:** Logistic Regression, selected automatically based on the
highest F1-score on the test set (see the app for the live, dynamically
generated explanation).
