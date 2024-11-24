import pandas as pd

def preprocess_data(df): #, sample_df
    # Dropping records with null values in column 'AttendingPhysician'
    df.dropna(subset=['AttendingPhysician'], axis=0, inplace=True)

    # Convert multiple columns to datetime in a single step
    date_columns = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)
    
    df['NoOfDays_Admitted'] = ((df['DischargeDt'] - df['AdmissionDt']).dt.days)+1
    df['NoOfDays_Admitted']

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

    # Ensure columns match the sample dataset
    # df = df[sample_df.columns]
    # Standardize data (if applicable)
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(df)
    # return df_scaled