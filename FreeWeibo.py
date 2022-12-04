import streamlit as st


header = st.container()
dataset = st.container()
features = st.container()
model_Training = st.container()


with header:
	st.title('Welcome to my FreeWeibo Dataset')



import mysql.connector as connector
import pandas as pd
import pandas.testing as tm
import sqlalchemy as sql 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import json
from scipy import stats
from streamlit_echarts import st_echarts
	
    #mydb = connector.connect(host="192.168.1.8", database = 'FreeWeibo',user="leberkc", passwd="Italia!23",use_pure=True)
mydb = connector.connect(host="96.242.223.246", database = 'FreeWeibo',user="leberkc", passwd="Italia!23",use_pure=True)
query = "Select * from FreeWeiboPosts"
data = pd.read_sql(query,mydb)
st.dataframe(data)
mydb.close() #close the connection



data = data[data['time_created'].notna()]
data = data[data['User_name'].notna()]
option = {
    "xAxis": {
        "type": "category",
        "data": data['time_created'].dt.strftime('%y-%m-%d %h:%m:%s').to_list(),
    },
    "yAxis": {"type": "category"},
    "series": [{"data": data['User_name'].to_list(), "type": "line"}],
}
st_echarts(
    options=option, height="400px",
)



#chart_data = pd.DataFrame(
#    columns = ["censored","time_created","User_name"])
#st.bar_chart(chart_data)















