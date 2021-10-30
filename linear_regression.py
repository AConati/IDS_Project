import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

topicnized_df = pd.read_csv("DATA/prepared_data/topicnized_data.csv")
print(topicnized_df.columns)
print(topicnized_df.head())

# Categories
# ['Unnamed: 0', 'Unnamed: 0.1', 'stars', 'useful', 'funny', 'cool',
#        'text', 'name', 'city', 'state', 'postal_code', 'latitude', 'longitude',
#        'review_count', 'is_open', 'categories', 'OutdoorSeating',
#        'RestaurantsGoodForGroups', 'Alcohol', 'RestaurantsDelivery',
#        'RestaurantsReservations', 'BusinessAcceptsCreditCards', 'BikeParking',
#        'HasTV', 'BusinessParking', 'RestaurantsAttire', 'GoodForKids',
#        'Ambience', 'NoiseLevel', 'RestaurantsTakeOut',
#        'RestaurantsPriceRange2', 'ByAppointmentOnly', 'CoatCheck',
#        'DogsAllowed', 'WiFi', 'HappyHour', 'WheelchairAccessible', 'Caters',
#        'RestaurantsTableService', 'BusinessAcceptsBitcoin', 'GoodForMeal',
#        'Corkage', 'BYOBCorkage', 'GoodForDancing', 'BestNights', 'Music',
#        'BYOB', 'DriveThru', 'Smoking', 'AcceptsInsurance', 'HairSpecializesIn',
#        'RestaurantsCounterService', 'Open24Hours', 'AgesAllowed',
#        'DietaryRestrictions', 'year', 'preparedText', 'Topic0', 'Topic1',
#        'Topic2', 'Topic3', 'Topic4', 'mostRelatedTopic'],
      # dtype='object'

# X_values
x_topic_0 = topicnized_df['Topic0']
x_topic_1 = topicnized_df['Topic1']
x_topic_2 = topicnized_df['Topic2']
x_topic_3 = topicnized_df['Topic3']
x_topic_4 = topicnized_df['Topic4']

X = pd.concat([x_topic_0, x_topic_1, x_topic_2, x_topic_3, x_topic_4], axis=1)
y = np.array(topicnized_df['stars'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# Now let's visualize how the estimated linear regression model performed by comparing predicted values to actual values

import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual stars')
plt.ylabel('Predicted stars')
plt.title('Actual vs Predicted')

# Finally let's compare predicted and actual values in dataframe
pred_df = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred, 'Difference': y_test-y_pred})
pred_df.head(20)