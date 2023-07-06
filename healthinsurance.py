import pickle
import pandas as pd


class HealthInsurance:

    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler =              pickle.load(open( self.home_path + 'src/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler =                         pickle.load(open( self.home_path + 'src/age_scaler.pkl', 'rb'))
        self.vintage_scaler =                     pickle.load(open( self.home_path + 'src/vintage_scaler.pkl', 'rb'))
        self.target_encode_gender_scaler =        pickle.load(open( self.home_path + 'src/target_encode_gender_scaler.pkl', 'rb'))
        self.target_encode_region_code_scaler =   pickle.load(open( self.home_path + 'src/target_encode_region_code_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel_scaler =     pickle.load(open( self.home_path + 'src/fe_policy_sales_channel_scaler.pkl', 'rb'))



    def data_cleaning(self, df1):
        #rename columns
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 'vehicle_damage',
                    'annual_premium', 'policy_sales_channel', 'vintage']
        
        #rename
        df1.columns = cols_new

        return df1
    
    def feature_engineering(self, df2):
        #feature engineering

        #vehicle damage number
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)

        #vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_years' if x == '1-2 Year' else 'below_1_years')

        return df2
    
    def data_preparation(self, df3):

        #annual_premium
        df3['annual_premium'] = self.annual_premium_scaler.transform(df3[['annual_premium']].values)

        #age
        df3['age'] = self.age_scaler.transform(df3[['age']].values)

        #vintage
        df3['vintage'] = self.vintage_scaler.transform(df3[['vintage']].values)

        #gender - Frequency Encoding
        df3.loc[:, 'gender'] = df3['gender'].map(self.target_encode_gender_scaler)

        #region_encoder - Target Encoding
        df3.loc[:, 'region_code'] = df3['region_code'].map(self.target_encode_region_code_scaler)

        #vehicle_age - One Hot Encoding
        df3 = pd.get_dummies(df3, prefix='vehicle_age', columns=['vehicle_age'])

        #policy_sales_channel - Frequency Encoding
        df3.loc[:, 'policy_sales_channel'] = df3['policy_sales_channel'].map(self.fe_policy_sales_channel_scaler)

        #feature selection
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage',
                 'policy_sales_channel', 'previously_insured']
        
        return df3[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #model prediction
        pred = model.predict_proba(test_data)

        #join prediction into original data
        original_data['prediction'] = pred[:, 1]

        return original_data.to_json(orient='records', date_format='iso')        