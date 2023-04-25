# SPACESHIP-SURVIVAL-PREDICTIONS
Kaggle dataset project predicting passengers survival using Random Forrest Classifier
This project utilizes the Kaggle data set spaceship titanic.  Will predict passengers by passenger_Id that survivie the spaceship titanic and indicate whether they were transported or not, utilizing the designation of True or False and out put this into a sample_submission.csv file.
Two files are utilized test.csv and train.csv, both are opened in a pandas data frame, as well as the existing sample_submission.csv file
TRAIN_DF
train_df = columns of PassengerID, HomePlanet, CryoSleep, Cabin Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Name Transported and consist of 8963 entries.
train_df is viewed to see types of data  - all True/False designations are changed to 1/0
All columns except PassengerID, Name, and Transported have null values
Columns containing null values that are numerical values are filled with the mean
Columns HomePlanet and Destination which represent multiple object type values, have dummies created to turn values into numerical values and column names remaned.
A function is created to clean test_df in the same manner as train_df
test_df is cleaned and named text_X
sklearn model selection, train_test_split is imported 
column names of train_df are designated as X
column name of train_df.Transported is desginated as y
train_test_split is run designating a sample size of 33%
RandomForestClassifer is imported and object created, X_train, y_train are fit to the object and y_pred is predicted for X_test
sklearn metrics accuracy score is imported and run on y_test, y_pred indicating an accuracy of 78%
test_X is reindexed to X_train columns
t_pred is fit with the RandomForrestClassifier and predict test_X
submission is created in a pandas dataframe creating column PassengerID and filling values with test_X.PassengerId, and a column for Transported and filling with t_pred values
The numerical values for Transported are mapped to a True false value for the Transported column.
The resulting dataframe provides passenger Id numbers and a true / false indication of whether the passenger was transported from the ship
