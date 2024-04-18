import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#loading data
loan_data = pd.read_csv('./Data set/loan_sanction_train.csv')
loan_data.drop(columns=['Loan_ID'], inplace=True)
loan_data.dropna(inplace=True)
loan_data['Loan_Status'] = loan_data['Loan_Status'].map({'N':0, 'Y':1})
dummies_data = pd.get_dummies(loan_data)
boolean_entries = dummies_data.select_dtypes(include=bool).columns
dummies_data[boolean_entries] = dummies_data[boolean_entries].astype(int)

X = dummies_data.drop(columns=['Loan_Status'])
y = dummies_data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=4)

#model
model = RandomForestClassifier(n_estimators=300, random_state=4)
model.fit(X_train, y_train)




def main():
    st.title("Home Loan Prediction App")

    # User input fields

    Gender = st.radio("Gender", ['Male', 'Female'], index=None,)
    Married = st.radio("Married", ('Yes', 'No'), index=None,)
    Dependents = st.selectbox("Dependents", ('0', '1', '2', '3+'))
    Education = st.radio("Education", ('Graduate', 'Not Graduate'), index=None,)
    Self_Employed = st.radio("Self Employed", ('Yes', 'No'), index=None,)
    ApplicantIncome = st.number_input("Applicant Income", value=None, placeholder="Enter your income")
    CoapplicantIncome = st.number_input("Coapplicant Income", value=None, placeholder="Enter your co-applicant income")
    LoanAmount = st.number_input("Loan Amount", value=None, placeholder="Enter loan amount")
    Loan_Amount_Term = st.number_input("Loan Amount Term", value=None, placeholder="Enter loan amount term")
    Credit_History = st.number_input("Credit History", value=None, placeholder="Enter your credit history")
    Property_Area = st.selectbox("Property Area", ('Urban', 'Semiurban', 'Rural'))



    if st.button("Predict"):
        # Preprocess user inputs
        input_data = {
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Add missing columns and set their values to zero
        missing_columns = set(X_train.columns) - set(input_df.columns)
        for col in missing_columns:
            input_df[col] = 0

        # Reorder columns to match the model's training data
        input_df = input_df[X_train.columns]


        input_df = pd.get_dummies(input_df)
        boolean_cols = input_df.select_dtypes(include=bool).columns
        input_df[boolean_cols] = input_df[boolean_cols].astype(int)

        prediction = model.predict(input_df)
        # Display prediction result
        if prediction[0] == 1:
            st.success("Congratulations! Your loan is approved.")
        else:
            st.error("Sorry, your loan application is rejected.")



if __name__ == '__main__':
    main()