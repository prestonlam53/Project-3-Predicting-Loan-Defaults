import streamlit as st
import pandas as np
import numpy as np

st.title('Predicting Client Default')
st.markdown("Answer the questions below to determine what % chance a client will default")


messages = ["How many months has client delayed payment as of April 2005?",
            "How many months has client delayed payment as of June 2005?",
            "What was the clients highest bill amount?",
            "What was the clients lowest bill amount?",
            "What was the clients total limit balance?",
            "What was the clients average bill?",
            "What was the clients highest pay amount?",
            "What was the clients lowest pay amount?",
            "How much did the client pay in total?",
            "How old is the client?"]

pay_0 = st.slider(messages[0], -1,8)
pay_2 = st.slider(messages[1], -1,8)
highest_bill = st.slider(messages[2], 0, 300000)
lowest_bill = st.slider(messages[3], 0, 300000)
limit_bal = st.slider(messages[4], 10000,100000)
avg_bill_amt = st.slider(messages[5], 0, 100000)
highest_pay = st.slider(messages[6], 0, 100000)
lowest_pay = st.slider(messages[7], 0, 50000)
Total_Pay_Amt = st.slider(messages[8], 0, 700000)
age = st.slider(messages[9], 0, 100)


max_bill_to_cred_lim = highest_bill/limit_bal
adjusted_bill_var = (highest_bill-lowest_bill)/avg_bill_amt
adjusted_pay_variance = (highest_pay-lowest_pay)/avg_bill_amt
paid_limit_ratio = Total_Pay_Amt/limit_bal


inputs = [pay_0, pay_2, max_bill_to_cred_lim, adjusted_bill_var,
       Total_Pay_Amt, avg_bill_amt, adjusted_pay_variance, age,
       paid_limit_ratio]



import pickle
my_model = pickle.load(open("pickled_model2.p","rb"))
#my_scaler = pickle.load(open('scaler.pkl', 'rb'))
def predict_default(inputs, model=my_model):

    inputs = [float(i) for i in inputs]
    # # inputs into the model
    # input_df = [[flour_cups_prop, sugar_cups_prop]]
    # make a prediction
    #inputs = scalers.transform(inputs)
    input_array = np.asarray(inputs)

    input_array = input_array.reshape(1,-1)
    prediction = my_model.predict(input_array)
    prob = model.predict_proba(input_array)[:, 1]
    # return a message
    message_array = ["No Default",
                     "Default"]

    #return message_array[prediction[0]]
    return prob[0]*100
message = predict_default(inputs)
st.header(str(message)+'% Chance of Default')
