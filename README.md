# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.


## Neural Network Model

![image](https://github.com/user-attachments/assets/ed81faab-be6b-40cb-b2ed-50c33237c2c0)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.
<br>
<br>
<br>
<br>
<br>
<br>
## PROGRAM
### Name: R Guruprasad
### Register Number: 212222240033
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet=gc.open('stdata').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
print(df)

df=df.astype({'in':float,'out':float})
x=df[['in']].values
y=df[['out']].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)

from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler()
Scaler.fit(x_train)
x_train1=Scaler.transform(x_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

m1=Sequential([
Dense(units=9,activation='relu',input_shape=[1]),
Dense(units=9,activation='relu'),
Dense(units=9,activation='relu'),
Dense(units=1)])

m1.summary()

m1.compile(optimizer='rmsprop',loss='mse')

m1.fit(x_train1,y_train,epochs=2000)

history=m1.history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Training Loss vs Iterations')
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Training Loss')

plt.legend()

plt.show()

xtrain1=Scaler.transform(x_test)
m1.evaluate(xtrain1,y_test)

n=[[30]]
x=Scaler.transform(n)
m1.predict(x)


```
## Dataset Information

![image](https://github.com/user-attachments/assets/f9b7849e-646d-4391-8ff1-13b7bd01486c)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/d4ca7044-094c-43b2-a87b-d9a5d722cc3c)

### Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/527ca3d2-7f38-4442-a76e-dc94a844a400)

### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/e38b0636-fc96-4d09-981c-4ca8b9d8fe27)

## RESULT

A neural network regression model for the given dataset has been developed sucessfully.
