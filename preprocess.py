import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

major_categories_path = 'data\\major_categories.pkl'
target_mean_dict_path = "data\\target_mean_dict.pkl"

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, major_categories_path):
        self.major_categories_path = major_categories_path
        self.major_categories = None

    def fit(self, X,y):
        # Load major categories during fit
        with open(self.major_categories_path, 'rb') as file:
            self.major_categories = pickle.load(file)
        return self

    def transform(self, X):
        df = X.copy()

        # Convert date columns to datetime
        date_columns = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])

        # Create 'NoOfDays_Admitted' column
        df['NoOfDays_Admitted'] = ((df['DischargeDt'] - df['AdmissionDt']).dt.days) + 1
        df.loc[df['IPD_OPD'] == 'OPD', 'NoOfDays_Admitted'] = 0

        # Create 'Age' column
        df['DOB'] = pd.to_datetime(df['DOB'], format='%Y-%m-%d')
        df['DOD'] = pd.to_datetime(df['DOD'], format='%Y-%m-%d', errors='coerce')
        df['Age'] = round(((df['DOD'] - df['DOB']).dt.days) / 365)
        df['Age'].fillna(round(((pd.to_datetime('2009-12-01') - df['DOB']).dt.days) / 365), inplace=True)

        # Create 'WhetherDead' column
        df['WhetherDead'] = df['DOD'].notna().astype(int)

        # Handle missing values in physician columns
        df[['OperatingPhysician', 'OtherPhysician']] = df[['OperatingPhysician', 'OtherPhysician']].fillna('No Physician')

        # Convert 'ClmDiagnosisCode_1' to string
        df['ClmDiagnosisCode_1'] = df['ClmDiagnosisCode_1'].astype(str)

        # Replace chronic condition values
        chronic_columns = [
            'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
            'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
            'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
            'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
        ]
        df[chronic_columns] = df[chronic_columns].replace(2, 0)

        # Replace 'RenalDiseaseIndicator' values
        df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].replace({'Y': 1}).fillna(0)

        # Replace categorical values
        df['IPD_OPD'].replace({'IPD': 1, 'OPD': 0}, inplace=True)
        df['Gender'].replace({1: 1, 2: 0}, inplace=True)

        # Handle mean encoding for columns
        mean_enc_columns = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1']
        for col in mean_enc_columns:
            df[col] = df[col].apply(lambda x: x if x in self.major_categories else 'Other')

        return df

class MeanEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, columns_to_encode):
        self.target_column = target_column
        self.columns_to_encode = columns_to_encode
        self.mean_encodings = {}

    def fit(self, X, y):
        # Compute mean encodings for each column to encode
        for col in self.columns_to_encode:
            self.mean_encodings[col] = y.groupby(X[col]).mean().to_dict()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns_to_encode:
            print(self.mean_encodings[col])
            X_transformed[col] = X[col].map(self.mean_encodings[col])
        return X_transformed

# Define a preprocessing step to drop unused columns
class ColumnSelector:
    def __init__(self, columns_to_keep):
        self.columns_to_keep = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns_to_keep]


# Not used in current version

def preprocess_data(df): 
    with open(major_categories_path, 'rb') as file:
        major_categories = pickle.load(file)

    with open(target_mean_dict_path, 'rb') as file:
        target_mean_dict = pickle.load(file)

    # Dropping records with null values in column 'AttendingPhysician'
    df.dropna(subset=['AttendingPhysician'], axis=0, inplace=True)

    # Convert multiple columns to datetime in a single step
    date_columns = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
    # print(df.columns)
    for col in date_columns:
        df[col] = df[col].apply(pd.to_datetime)    
    
    df['NoOfDays_Admitted'] = ((df['DischargeDt'] - df['AdmissionDt']).dt.days)+1

    # OPD patients admission days will be 0
    df.loc[df['IPD_OPD']=='OPD', 'NoOfDays_Admitted']  = 0

    # Dropping date variables
    df.drop(['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt'], axis=1, inplace = True)

    ## Lets Create Age column to the dataset
    df['DOB'] = pd.to_datetime(df['DOB'], format = '%Y-%m-%d')
    df['DOD'] = pd.to_datetime(df['DOD'], format = '%Y-%m-%d', errors='ignore')

    df['Age'] = round(((df['DOD'] - df['DOB']).dt.days)/365)
    df.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - df['DOB']).dt.days)/365),
                                    inplace=True)
    
    df.loc[df.DOD.isna(),'WhetherDead']= 0
    df.loc[df.DOD.notna(),'WhetherDead']= 1

    # df.drop(['OperatingPhysician','OtherPhysician'], axis=1, inplace= True)
    
    df[['OperatingPhysician','OtherPhysician']] = df[['OperatingPhysician','OtherPhysician']].fillna('No Physician')

    df['ClmDiagnosisCode_1'] = df['ClmDiagnosisCode_1'].astype('str')

    df = df.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
            'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
            'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
            'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    df = df.replace({'RenalDiseaseIndicator': 'Y'}, 1)
    df.RenalDiseaseIndicator.replace(['Y','0'],[1,0],inplace=True)
    # df.PotentialFraud.replace(['Yes','No'],[1,0],inplace=True)
    df['IPD_OPD'].replace(['IPD','OPD'],[1,0],inplace=True)
    df['Gender'].replace([1,2],[1,0],inplace=True)

    # physician_cols = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']

    # df.drop(physician_cols,axis=1,inplace=True)   

    # with open('major_categories.pkl', 'rb') as file:
    #     major_categories = pickle.load(file)

    # categorical_column = df.select_dtypes('O').columns

    # prev categorical_column
    mean_enc_columns = ['AttendingPhysician','OperatingPhysician','OtherPhysician','ClmDiagnosisCode_1']
    
    for col in mean_enc_columns:
        df[col] = df[col].apply(lambda x: x if x in major_categories else 'Other')

    # Categories with less than 1% of records to "Other"

    # Target encoding for each categorical column
    for col in mean_enc_columns:
        mean_encoding = target_mean_dict[col]
        # Apply target encoding to both train and test sets
        df[f'{col}_encoded'] = df[col].map(mean_encoding)


    return df, mean_enc_columns

    # Ensure columns match the sample dataset
    # df = df[sample_df.columns]
    # Standardize data (if applicable)
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df)
    # return df_scaled