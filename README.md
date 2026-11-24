# Predicting Tech Salaries from Glassdoor Job Postings

An end-to-end data science project (EDA + regression) built as an AlmaBetter capstone. It uses Glassdoor job postings from 2017-2018 to understand what drives salary in data and analytics roles, and to predict the average annual salary of a posting.

**Notebook:** `Glassdoor_Salary_Prediction_Submission.ipynb`

**Colab link:** https://colab.research.google.com/drive/1ePZrg6TfZQ7Yz72aGMacFvEXf1Tf2Jlu?usp=sharing


---

## Table of contents

- [Business problem](#business-problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Key insights](#key-insights)
- [Model results](#model-results)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [Tech stack](#tech-stack)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Author](#author)

---

## Business problem

Salary information in tech is scattered and opaque. Job seekers do not know what to ask for, employers do not know what the market pays, and recruiters need benchmarks for fair offers. This project answers four questions:

1. How does salary vary by job position (Data Scientist vs Data Analyst vs Data Engineer)?
2. What is the impact of company size on salary?
3. How do salaries differ by location (for example San Francisco vs New York)?
4. Can a model predict the salary of a posting from its attributes?

## Dataset

- **Source:** Glassdoor job postings, 2017-2018 (`glassdoor_jobs.csv`)
- **Size:** 956 rows x 15 columns (job title, salary estimate, job description, rating, company, location, headquarters, size, founded year, ownership, industry, sector, revenue, competitors)
- **Target:** `avg_salary`, the midpoint of the salary range in $K per year

**Data quality issues found and fixed**

| Issue | How it was handled |
|---|---|
| Missing values hidden as `-1` / `Unknown` (e.g. 66% of Competitors, 32% of Revenue) | Converted to real `NaN`, then imputed inside the model pipeline |
| 356 duplicate rows (37%) | Removed before splitting to avoid train/test leakage |
| Salary stored as text (`$53K-$91K`, `$21-$34 Per Hour`) | Parsed to numbers; hourly wages converted to annual using 2,080 hours |
| 214 rows with no salary | Dropped (the target cannot be imputed honestly) |

After cleaning, **467 unique salaried postings** remain.

## Approach

1. **Data understanding and wrangling:** reusable helper functions to clean, parse and engineer features.
2. **Feature engineering:** 6 role families and 5 seniority levels from 328 raw job titles, state, company age, number of competitors, ordinal company size, description length, and 11 skill flags mined from the job description (Python, SQL, R, Spark, AWS, Excel, Tableau, Hadoop, TensorFlow, machine learning, PhD).
3. **EDA:** 15 visualisations (univariate, bivariate and multivariate) with business insights for each.
4. **Hypothesis testing:** Welch's t-test, Kruskal-Wallis and Mann-Whitney U.
5. **Preprocessing:** outlier handling with the IQR rule, ordinal and one-hot encoding, a full text pipeline (cleaning, stemming, TF-IDF, Truncated SVD), feature selection, and an 80/20 split.
6. **Modelling:** Ridge Regression, Random Forest and Gradient Boosting inside leakage-safe pipelines, tuned with GridSearchCV / RandomizedSearchCV (5-fold CV), compared with a mean-prediction baseline.
7. **Explainability:** permutation importance on the unseen test set.

## Key insights

- **Role is the biggest driver.** Data scientists average about $117K, data analysts about $69K (Welch t-test, p < 0.001).
- **Seniority adds steady pay:** about $68K (junior), $92K (mid), $117K (senior), $137K (lead).
- **Location matters.** California postings have a median of $120.5K against $92.2K elsewhere (Mann-Whitney, p < 0.001). San Francisco (about $134K) pays roughly $29K more than New York (about $105K).
- **Company size has a small, U-shaped effect.** Very small and very large firms pay more than mid-sized ones (Kruskal-Wallis, p = 0.022).
- **Skills carry a premium:** machine learning (+$27K), PhD (+$25K), Python (+$20K), TensorFlow (+$18K). Excel (-$20K) and Tableau (-$8K) go with lower pay, mainly because they appear in analyst roles (correlation, not causation).
- **Company rating, age and description length** have almost no relationship with salary.

## Model results

Test set = 94 postings. Training set = 373 postings. Salaries are in $K.

| Model | Test R2 | Test MAE | Test RMSE | CV MAE (5-fold) |
|---|---|---|---|---|
| Baseline (predict the mean) | -0.02 | 31.03 | 40.57 | - |
| Ridge (tuned) | 0.440 | 22.01 | 30.12 | 19.68 |
| Random Forest (tuned) | 0.419 | 21.83 | 30.66 | 19.67 |
| Gradient Boosting (tuned) | 0.437 | 22.25 | 30.18 | 19.68 |

- **Chosen model:** Random Forest (tuned), selected by the lowest cross-validated MAE rather than by the test set.
- **Honest note:** the three tuned models are practically tied. Ridge is an equally good choice and easier to explain.
- **Top drivers of the predictions** (permutation importance): analyst role, California location, mid-level seniority, machine-learning requirement and data-scientist role. Company size, rating and age add very little.
- Text features (TF-IDF + SVD) were tested and did not improve cross-validated error, so they were left out of the final model.

> Exact numbers can differ slightly between library versions.

## Project structure

```
.
├── Glassdoor_Salary_Prediction_Submission.ipynb   # full analysis and modelling notebook
├── glassdoor_jobs.csv                             # dataset
├── best_salary_model.joblib                       # saved model (created when the notebook runs)
├── model_feature_columns.joblib                   # feature order for predictions (created by the notebook)
└── README.md
```

## How to run

**Option 1: Google Colab (easiest)**

1. Open the notebook in Colab.
2. Run the cells in order (Runtime, then Run all).
3. When asked, upload `glassdoor_jobs.csv`.

**Option 2: locally**

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib jupyter
jupyter notebook Glassdoor_Salary_Prediction_Submission.ipynb
```

Keep `glassdoor_jobs.csv` in the same folder as the notebook.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, Joblib.

## Limitations

- Small sample (467 postings, 94 test rows), so results are noisy.
- Data is from 2017-2018, and Glassdoor values are estimates rather than actual pay.
- No cost-of-living data, so higher pay in a city does not mean higher purchasing power.
- Postings from the same company may appear in both training and test sets.
- Some groups (for example ML Engineers and Austin) have too few postings for reliable estimates.

## Future work

- Collect larger, more recent salary data and add experience and education.
- Try text embeddings for the job description.
- Deploy the model as a Streamlit app (optionally with the Gemini API for natural-language questions).

## Author

**Surya Rohila**
www.linkedin.com/in/surya-rohila-493779110
