import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
scaler = StandardScaler()
le = LabelEncoder()

# Reading the dataset 

df = pd.read_csv('Dataset.csv')

# Now filtering the data and dropping the missing values

df.dropna(inplace=True)

# Data processing is done as we cannot treat outliers because it may affect the user convinience if we remove or change the details

""" 

Dropping the unwanted columns in the database 
For your convinience you may use Python notebook
Restaurant Name, Restaurant ID, Country Code, City, Address, Locality, Locality Verbose, Longitude, Latitude
Currency, Has Table booking, Has online delivery, Is delivering now, Switch to order menu

Total columns/features to drop : 14
"""

df_final = df.drop(columns=['Restaurant Name', 'Restaurant ID', 'City', 'Address', 'Locality', 'Locality Verbose', 'Country Code', 'Longitude', 'Latitude', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu'])

# Now we will work only with df_final data frame
# Label Encoding the Rating Color and Text because aggregate Rating is mainly dependent on them 
# And converting the strings to numerical data

rate_text = le.fit_transform(df_final['Rating text'])
rate_color = le.fit_transform(df_final['Rating color'])
cuisines = le.fit_transform(df_final['Cuisines'])
vote = scaler.fit_transform(df_final[['Votes']])

# After transforming the data replacing it with the data frame opened

df_final['Rating color'] = rate_color
df_final['Rating text'] = rate_text
df_final['Cuisines'] = cuisines
df_final['Votes'] = vote

# Creating the final Input and Output Dataset

X = df_final.drop(columns='Aggregate rating')
y = round(df['Aggregate rating'])

# round function is used because the model will treat the aggregate rating as continuous values and we are giving classifier values 
# it will throw an error if we don't round off the values

# Creating train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Now creating a model and fitting the data in the model

model = RandomForestClassifier(n_estimators=300, criterion='log_loss')

model.fit(X_train, y_train)

print(model.score(X_test, y_test))