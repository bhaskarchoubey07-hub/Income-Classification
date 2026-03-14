# Income Classification Dashboard

This project is a Streamlit dashboard for exploring census-style income data and predicting whether a person's income is `>50K` or `<=50K` with a Random Forest classifier.

## Features

- Sidebar for dataset upload, dataset info, and feature selection
- Data overview with preview, shape, missing values, and column types
- Interactive Plotly visualizations for key distributions and relationships
- Random Forest training pipeline with preprocessing and evaluation metrics
- Feature importance chart
- Prediction panel for quick income classification

## Project Structure

```text
income-dashboard/
|-- app.py
|-- model.py
|-- data/
|   `-- income_evaluation.csv
|-- requirements.txt
`-- README.md
```

## Dataset

The app expects a CSV file named `income_evaluation.csv` with these columns:

- `age`
- `workclass`
- `fnlwgt`
- `education`
- `education-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`
- `income`

You can either:

- use the bundled starter dataset in `data/income_evaluation.csv`
- upload your own CSV from the Streamlit sidebar

## Run

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```
