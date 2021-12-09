#!/usr/bin/env python
# coding: utf-8

# ## Data Visualization

import csv
import json
# from shapely.geometry import Point, Polygon
import geopandas as gpd
import numpy as np 
import urllib.request as ur 
import pandas as pd 
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
import streamlit as st
import pydeck as pdk
from urllib.error import URLError
import plotly.express as px

# ## Streamlit
# create the slider and checkbox
st.set_page_config(layout="wide")

st.write("""
# RetroScope
### Kendall Square Retrofit
This app predicts the energy, carbon emissions, cost effectiveness and health improvement of different retrofitting strategies
""")
st.sidebar.header('User Input Parameters')
st.sidebar.markdown('#### Select percentage of different building types you want to renovate')

office = st.sidebar.slider('Office', 0.0, 1.0, step=0.1)
housing = st.sidebar.slider('Housing', 0.0, 1.0, step=0.1) 
lab = st.sidebar.slider('Laboratory', 0.0, 1.0, step=0.1)


st.sidebar.markdown('#### Retrofitting Level')
retrofit = ['Base Case', 'Conventional', 'Deep Energy', 'Deep Energy Plus']
RL = st.sidebar.radio('Select the retrofit level', retrofit, index=1)


#########################################################
#creat the new dataframe based on selection

# df=pd.read_csv('C:/Users/Elence/OneDrive - Harvard University/Desktop/City Science/Project/Data/kendall_new3.csv')
df=pd.read_csv('https://raw.githubusercontent.com/ElenceChen/Kendall_Retrofit/main/Kendall_new3.csv')

raw_4 = []
for i in range(len(df.geometry)):
    p_str = df.geometry[i]
    raw_1 = p_str.split(',')
    raw_1[0] = raw_1[0].split("(")[-1]
    raw_1[-1] = raw_1[-1].split(")")[0]
    raw_2 = []
    for j in range(len(raw_1)):
        temp = raw_1[j].split(" ")
        raw_2.append([float(temp[-2]), float(temp[-1])])
    raw_3 = [raw_2]
    raw_4.append(raw_3)
df["coordinates"] = raw_4

### Calculate % Carbon reduction, Cost efficiency 
# CR: Carbon Reduction (0-1)
# CE: Cost Efficiency (0-1)
# CRet: Carbon after retrofit
# TCR: total carbon reduction (tons of CO2)
RetroLe = ['base','con', 'DE', 'DEP']

def carbon_reduction_per(Cbase, CRet):
    return (Cbase-CRet)/Cbase

def cost_efficiency(COST, A, Cbase,CRet):
    return (1000000*(Cbase-CRet)/A)/COST

def tot_carbon_reduction(Cbase, CRet):
    return Cbase-CRet

for i in RetroLe:
    if i == 'base':
        df['CR_'+ i] = np.zeros(df.shape[0])
        df['CE_'+ i] = np.zeros(df.shape[0])
        df['TCR_'+ i] = np.zeros(df.shape[0])
    else: 
        Cbase = df.CO2e_base
        CRet = df['CO2e_'+i]
        df['CR_'+ i] = carbon_reduction_per(Cbase, CRet)
        COST = df['cost_' + i]
        A = df.area
        df['CE_'+ i] = cost_efficiency(COST, A, Cbase, CRet) 
        df['TCR_'+ i] = tot_carbon_reduction(Cbase, CRet)

cols_to_norm = ['CE_con','CE_DE', 'CE_DEP']
df[['CE_con_norm', 'CE_DE_norm', 'CE_DEP_norm']] = df[cols_to_norm].apply(lambda x: x / x.max())
df['CE_base_norm'] = np.zeros(df.shape[0])
df.head()


#### create selected dataframes for building types

df_office = df[df.type == 'Office'].sort_values(['vintage'], ascending=False)
df_housing = df[df.type == 'Housing'].sort_values(['vintage'], ascending=False)
df_lab = df[df.type == 'Laboratory'].sort_values(['vintage'], ascending=False)

office_sel_per = df_office.head(round(office*len(df_office)))
housing_sel_per = df_housing.head(round(housing*len(df_housing)))
lab_sel_per = df_lab.head(round(lab*len(df_lab)))

frames = [office_sel_per, housing_sel_per, lab_sel_per]
df_new = pd.concat(frames)

##### converter 
converter = {'Base Case': 'base', 'Conventional':'con', 'Deep Energy':'DE', 'Deep Energy Plus':'DEP'}
eui = 'EUI_' + converter[RL]
co2e = 'CO2e_' + converter[RL]
health = 'health_' + converter[RL]
carbon_redu = 'CR_' + converter[RL]
tot_carbon_redu = 'TCR_' + converter[RL]
cost_eff = 'CE_'+ converter[RL]
cost_eff_norm = 'CE_'+ converter[RL] + '_norm'

df_eui = df_new[['coordinates', 'lat', 'lon', eui, co2e,'type', 'bldg_ht']]
df_co2e = df_new[['coordinates', 'lat', 'lon', co2e]]
df_health = df_new[['coordinates','lat', 'lon', health]]
df_cost_eff = df_new[['coordinates','lat', 'lon', cost_eff]]


# df_eui.to_json (r'C:/Users/Elence/eui.json', orient = 'records')
df_co2e.to_json (r'C:/Users/Elence/co2e.json', orient = 'records')
df_health.to_json (r'C:/Users/Elence/health.json', orient = 'records')

file_eui = 'C:/Users/Elence/eui.json'
file_co2e = 'C:/Users/Elence/co2e.json'
file_health = 'C:/Users/Elence/health.json'

#########################################################
# Custom color scale
#Green to red
COLOR_RANGE = [
    [0,255,0],
    [116,186,0],
    [231,243,0],
    [255,244,0],
    [255,229,0],
    [255,215,0],
    [255,201,0],
    [255,186,0],
    [255,172,0],
    [255,158,0],
    [255,143,0],
    [255,129,0],
    [255,115,0],
    [255,100,0],
    [255,86,0],
    [255,72,0],
    [255,57,0],
    [255,43,0],
    [255,29,0],
    [255,14,0],
    [255,0,0],
]


# COLOR_RANGE = [[16, 39, 123],
#  [0, 69, 142],
#  [0, 88, 123],
#  [0, 101, 82],
#  [24, 109, 38],
#  [24, 109, 40],
#  [49, 124, 44],
#  [72, 139, 48],
#  [94, 155, 52],
#  [118, 170, 55],
#  [142, 185, 59],
#  [167, 200, 63],
#  [193, 215, 67],
#  [220, 230, 73],
#  [248, 244, 79],
#  [255, 242, 75],
#  [251, 225, 64],
#  [245, 208, 55],
#  [239, 191, 47],
#  [232, 175, 40],
#  [224, 158, 35],
#  [216, 142, 30],
#  [206, 127, 27],
#  [196, 112, 25],
#  [185, 97, 23],
#  [174, 82, 22]]


COLOR_RANGE2 = COLOR_RANGE[::-1]

carbon = np.linspace(15, 2488, num=21).tolist()
health_color = np.linspace(0, 100, num=21).tolist()


def color_scale(val):
    for i, b in enumerate(carbon):
        if val < b:
            return COLOR_RANGE[i]
    return COLOR_RANGE[i]

def color_scale2(val):
    for i, b in enumerate(health_color):
        if val < b:
            return COLOR_RANGE2[i]
    return COLOR_RANGE2[i]

def calculate_elevation(val):
    return math.pow(val, 2)

df_eui["fill_color"] = df_eui[co2e].apply(lambda x: color_scale(x))
df_health["fill_color"] = df_health[health].apply(lambda x: color_scale2(x))


#########################################################
row1_1, row1_2 = st.columns((2,1))

with row1_1:
    st.write('**Total Carbon Emissions**')

#Plot 1st Maps - Carbon & EUI

    try:
        ALL_LAYERS = {
            'Original Building Height': pdk.Layer(
                'PolygonLayer',
                df_eui,
                opacity=0.7,
                stroked=True,
                get_position=["lon", "lat"], 
                get_polygon="coordinates",
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation='bldg_ht',
                get_fill_color= [255,255,255],
                get_line_color=[0,0,0],
                auto_highlight=False,
                pickable=True
            ), 

            'Total GHG Emissions': pdk.Layer(
                'PolygonLayer',
                df_eui,
                # id="geojson", ######################
                opacity=0.7,
                stroked=True,
                get_position=["lon", "lat"], 
                get_polygon="coordinates",
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation=eui,
                get_fill_color= 'fill_color',
                get_line_color='fill_color',
                auto_highlight=False,
                pickable=True
            ), 

            "GHGe Radius": pdk.Layer(
                "ScatterplotLayer",
                data=pd.read_json(file_co2e),
                get_position=["lon", "lat"],
                get_color=[130, 156, 181],
                get_radius=co2e,
                opacity=0.3,
                radius_scale=0.15)
        }
        st.sidebar.markdown('### GHG Emissions Map')
        selected_layers = [
            layer for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)]
        if selected_layers:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": 42.36, 
                                    "longitude": -71.085, "zoom": 13, "pitch": 50},
                layers=selected_layers,
            ))


        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error("""
            **This demo requires internet access.**

            Connection error: %s
        """ % e.reason)

############################################################################
# legend 
from PIL import Image
image = Image.open('C:/Users/Elence/OneDrive - Harvard University/Desktop/City Science/Project/Code/colorbar2.png')
st.image(image, use_column_width=True)

#############################################################################

#Plot 2nd Maps - Cost & Health

color2 = [[123, 24, 16],
[148, 70, 33],
[171, 111, 57],
[192, 151, 90],
[213, 191, 129],
[235, 231, 173]]


CE_color = np.linspace(20, 280, num=6).tolist()

def color_scale3(val):
    for i, b in enumerate(CE_color):
        if val < b:
            return color2[i]
    return color2[i]

df_cost_eff["fill_color"] = df_cost_eff[cost_eff].apply(lambda x: color_scale3(x))


with row1_2:
    st.write('**Health Improvement & Cost Efficiency**')

    try:
        ALL_LAYERS = {
            "Health Improvement": pdk.Layer(
                #"HexagonLayer",
                'ColumnLayer',
                df_health,
                get_position=["lon", "lat"],
                get_elevation=health,
                opacity=0.2,
                radius=30, 
                elevation_scale=10,
                elevation_range=[50, 250],
                extruded=True,
                pickable=True,
                # coverage =1,
                get_fill_color='fill_color', 
                get_line_color='fill_color'
                ),


            "Cost Efficiency": pdk.Layer(
            'ScatterplotLayer',     
            df_cost_eff,
            get_position=['lon', 'lat'],
            auto_highlight=True,
            get_radius=cost_eff, 
            opacity=0.3,         
            get_fill_color= 'fill_color',
            pickable=True)

        }


        st.sidebar.markdown('### Health & Cost Map')
        selected_layers = [
            layer for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)]
        if selected_layers:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": 42.36, 
                                    "longitude": -71.085, "zoom": 12.5, "pitch": 50},
                layers=selected_layers,
            ))


        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error("""
            **This demo requires internet access.**

            Connection error: %s
        """ % e.reason)

#################################################################
# total carbon reduction potential based on the selection
TCR_dic = {}
Of = []
Hou = []
La = []
for i in range(len(df_new)):
    if df_new.type.iloc[i] == 'Office':
        Of.append(df_new[tot_carbon_redu].iloc[i])
    if df_new.type.iloc[i]  == 'Housing':
        Hou.append(df_new[tot_carbon_redu].iloc[i])
    if df_new.type.iloc[i]  == 'Laboratory':
        La.append(df_new[tot_carbon_redu].iloc[i])
        
TCR_dic['Office'] = int(sum(Of))
TCR_dic['Housing'] = int(sum(Hou))
TCR_dic['Laboratory'] = int(sum(La))

df_TCR = pd.DataFrame(TCR_dic, index= ['CO2e Reduction(tons)'])
df_TCR['Total'] = df_TCR.sum(axis=1)


#################################################################
# Plot Radar Chart & TCR dataframe
import plotly.graph_objects as go

st.write('**Radar Chart**')

categories = ['Percentage of Carbon Reduction','Cost Efficiency','Improvement of Health', 'Percentage of Carbon Reduction']
H1 = df_new[df_new.type == 'Housing'] 
O1 = df_new[df_new.type == 'Office'] 
L1 = df_new[df_new.type == 'Laboratory'] 

# r = [df_new[carbon_redu].mean(), df_new[cost_eff_norm].mean(), df_new[health].mean()/100, df_new[carbon_redu].mean()] 

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r = [H1[carbon_redu].mean(), H1[cost_eff_norm].mean(), H1[health].mean()/100, H1[carbon_redu].mean()] ,
        theta = categories,
#         fill='toself',
#         fillcolor='#b8b8b8',
#         line_color='#b8b8b8',
        line_width=2,
        opacity=0.8,
        name = 'Housing',
        showlegend=True
    ))


fig.add_trace(go.Scatterpolar(
      r=[O1[carbon_redu].mean(), O1[cost_eff_norm].mean(), O1[health].mean()/100, O1[carbon_redu].mean()] ,
      theta=categories,
#       fill='toself',
#       fillcolor='#9cdbe7',
#       line_color='#9cdbe7',
      line_width=2,
      opacity=1,
      name='Office',
     showlegend=True
))

fig.add_trace(go.Scatterpolar(
      r=[L1[carbon_redu].mean(), L1[cost_eff_norm].mean(), L1[health].mean()/100, L1[carbon_redu].mean()] ,
      theta=categories,
#       fill='toself',
#       fillcolor="#E4FF87",
#       line_color="#E4FF87",
      line_width=2,
      opacity=0.4,
      name='Lab',
      showlegend=True
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
)

st.write(fig)



st.write('**Carbon Reduction Potential**')
st.dataframe(df_TCR)

