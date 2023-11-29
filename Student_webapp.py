import streamlit as st
import pickle
from PIL import Image

# Load the pickled model
with open('modelrf.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit web app
st.header("Student performance")
image = Image.open('per.png')

st.image(image)
st.sidebar.header("This is a model to predict student's performance")

Hours_Studied=st.sidebar.slider("Select hours studied", 0, 10, 5)
prev_score=st.sidebar.slider("Select previous score", 0, 100, 50)
Sleep=st.sidebar.slider("Select sleep hours ", 0, 10, 5)
practise=st.sidebar.selectbox("Sample papers practised",[0,1,2,3,4,5,6])

X_test = [[Hours_Studied , prev_score, Sleep,practise]]
st.write("Hours studied is:", Hours_Studied)
st.write("Previous score is:", prev_score)

st.write("sleep hours is:", Sleep)
st.write("Sample papers practised is:", practise)


yhat_test = model.predict(X_test)


st.subheader("Performance index is:" )
st.write(yhat_test)