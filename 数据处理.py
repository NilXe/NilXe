import pandas as pd
from sklearn import preprocessing

p = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


p['gender'] = p['gender'].map({'Male': 0, 'Female': 1})
p['SeniorCitizen'] = p['SeniorCitizen']
p['Partner'] = p['Partner'].map({'Yes': 1, 'No': 0})
p['Dependents'] = p['Dependents'].map({'Yes': 1, 'No': 0})

tenure = []
p['tenure'] = preprocessing.scale(p['tenure'])

p['PhoneService'] = p['PhoneService'].map({'Yes': 1, 'No': 0})
p['MultipleLines'] = p['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 2})
p['InternetService'] = p['InternetService'].map({'DSL': 1, 'No': 0, 'Fiber optic': 2})
p['OnlineSecurity'] = p['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['OnlineBackup'] = p['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['DeviceProtection'] = p['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['TechSupport'] = p['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['StreamingTV'] = p['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['StreamingMovies'] = p['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': 2})
p['Contract'] = p['Contract'].map({'Month-to-month': 0, 'One year': 1})
p['PaperlessBilling'] = p['PaperlessBilling'].map({'Yes': 1, 'No': 0})
p['PaymentMethod'] = p['PaymentMethod'].map({'Electronic check': 1, 'Mailed check': 0, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
p['MonthlyCharges'] = preprocessing.scale(p['MonthlyCharges'])

# print(p['TotalCharges'], type(p['TotalCharges'][0]))
charges = []
for i in p['TotalCharges']:
    if i == ' ':
        print('-------------------------')
        i = 0
        charges.append(i)
    else:
        i = float(i)
        charges.append(i)

p['TotalCharges'] = preprocessing.scale(charges)
p['Churn'] = p['Churn'].map({'Yes': 1, 'No': 0})

p = p.dropna()
p.to_csv('train_data.csv', index=False)
print('over')
