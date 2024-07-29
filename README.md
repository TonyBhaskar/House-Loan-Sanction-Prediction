# House-Loan-Sanction-Prediction

A web application that predicts whether a home loan will be sanctioned based on user input data.

## Overview

This application uses a Random Forest Classifier model trained on a dataset of loan sanction data to predict the likelihood of a loan being sanctioned. The model takes into account various factors such as applicant income, co-applicant income, loan amount, credit history, and property area.

## Features

* User-friendly interface for inputting data
* Predicts loan sanction outcome based on user input data
* Uses a Random Forest Classifier model trained on a dataset of loan sanction data

## Requirements

* Python 3.8+
* Streamlit 0.85+
* Scikit-learn 0.24+
* Pandas 1.2+

## Usage

1. Clone the repository: `git clone https://github.com/tonybhaskar/House-Loan-Sanction-Prediction.git`
2. Install the required libraries
3. Run the application: `streamlit run app.py`
4. Input your data and click the "Predict" button to see the outcome

## Dataset

The dataset used to train the model is included in the `Data set` folder. It contains 614 rows of data with 13 columns.

## Model

The model used is a Random Forest Classifier with 300 estimators and a random state of 4.

## Contributing

Contributions are welcome! If you'd like to improve the model or add new features, please submit a pull request.

## License

This project is licensed under the MIT License.
