from sqlalchemy import true
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from vega_datasets import data  
import plotly.express as px




st.set_page_config(layout="wide")


df = pd.read_csv('/Users/juniorbenitez/Documents/main project/personal_project/debug_validation/telemetry_timeseries.csv')

df2 = pd.read_csv('/Users/juniorbenitez/Documents/main project/personal_project/debug_validation/match_summary.csv')

utility_df= df[
    [
    "hill_number",
    "nades_used_to_that_point",
    "stuns_used_to_that_point",
    "specialties_used_to_that_point"
    ]
    ].copy()

utility_df["nades_delta"] = utility_df["nades_used_to_that_point"].diff()
    

utility_df["stuns_delta"] =utility_df["stuns_used_to_that_point"].diff()
    

utility_df["specialties_delta"] = utility_df["specialties_used_to_that_point"].diff()

per_hill_df = (
    utility_df.groupby("hill_number", as_index=False)
    [
        ["nades_delta","stuns_delta","specialties_delta"
        ]
    ].sum().sort_values("hill_number")
    
    )


player_df = df2.iloc[0:4,0:9]
player_df = player_df.T
player_df.columns = player_df.iloc[0]
player_df = player_df.iloc[1:]

player_df.index.name="player_name"
numeric_cols=["kd","total_kills","total_deaths"]
player_df[numeric_cols] = player_df[numeric_cols].apply(pd.to_numeric, errors="coerce")


grouped_df = player_df.groupby("team")[numeric_cols].sum()
















#title_container = st.container(key="title", width="stretch", height="content")
#with title_container:
    #st.title("ESPORTS DASHBOARD", text_alignment="center")


filter_bar = st.sidebar
with filter_bar:
    st.title("Sidebar filters will go here.")





kpi_container = st.container(key="kpis", width="stretch", height = 140,border=False)
with kpi_container:
    col_1, col_2, col_3, col_4, col_5 = st.columns(5,gap="small",border=True)  
    with col_1:

        player_name=st.selectbox( label = "name",options = df2.columns[1:9])
    
    with col_2:
        st.write("KD")
        st.write(df2.loc[1, player_name])
    
    with col_3:
        st.write ("gernade usage")
        st.write(df2.loc[4, "all_match"])
    
    
    with col_4:
        st.write("tactical usage")
        tactical_usage = str(df2.loc[5, "all_match"])
        st.write(tactical_usage)
    
    with col_5:
        st.write("speciality usage")
        st.write(df2.loc[6, "all_match"])
        



chart_row_1 = st.container(key = "data" , width = "stretch")
with chart_row_1:
    chart_1_col_1, chart_3_col_3, chart_4_col_4 = st.columns([1.4,0.8,0.8])
    with chart_1_col_1:
        chart_5_col_1, chart_5_col_2 = st.columns([1,1])
        with chart_5_col_1:
            with st.container(key="ally_row_1", border=True,height = 100):
                col_player_1_image, col_player_1_kd = st.columns([0.7,1])
                with col_player_1_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/ally_player.png", width="content")

                with col_player_1_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[1]} </b><br> 
                        KD: {df2.iloc[1,1]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                
            with st.container(key="ally_row_2", border=True,height=100):
                col_player_2_image, col_player_2_kd = st.columns([0.7,1])
                with col_player_2_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/ally_player.png")
                with col_player_2_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[2]} </b><br> 
                        KD: {df2.iloc[1,2]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                    
            with st.container(key="ally_row_3", border=True,height=100):
                col_player_3_image, col_player_3_kd = st.columns([0.7,1], vertical_alignment="center")
                with col_player_3_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/ally_player.png")
                with col_player_3_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[3]} </b><br> 
                        KD: {df2.iloc[1,3]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)

            with st.container(key="ally_row_4", border=True,height =100):
                col_player_4_image, col_player_4_kd = st.columns([0.7,1])
                with col_player_4_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/ally_player.png")
                with col_player_4_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[4]} </b><br> 
                        KD: {df2.iloc[1,4]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                
                
                
                

        with chart_5_col_2:
            with st.container(key="enemy_row_1",height = 100):
                col_player_5_image, col_player_5_kd = st.columns([0.7,1])
                with col_player_5_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/Enemy_player.png")

                with col_player_5_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[5]} </b><br> 
                        KD: {df2.iloc[1,5]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                
            with st.container(key="enemy_row_2", border=True,height=100):
                col_player_6_image, col_player_6_kd = st.columns([0.7,1])
                with col_player_6_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/Enemy_player.png")
                with col_player_6_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[6]} </b><br> 
                        KD: {df2.iloc[1,6]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                    
            with st.container(key="enemy_row_3", border=True,height=100):
                col_player_7_image, col_player_7_kd = st.columns([0.7,1])
                with col_player_7_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/Enemy_player.png")
                with col_player_7_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[7]} </b><br> 
                        KD: {df2.iloc[1,7]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)

            with st.container(key="enemy_row_4", border=True,height =100):
                col_player_8_image, col_player_8_kd = st.columns([0.7,1])
                with col_player_8_image:
                    st.image("/Users/juniorbenitez/Documents/main project/personal_project/assets/images_dashboard/Enemy_player.png")
                with col_player_8_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[8]} </b><br> 
                        KD: {df2.iloc[1,8]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                
                
            
 
            

    with chart_3_col_3:
        with st.container(key= "chart_3",border=True,height= 400):

            st.bar_chart(per_hill_df,y='hill_number',x="stuns_delta")



            


    






    with chart_4_col_4:
        with st.container(key="chart_4",border=True):
            #uses parsed name to get total kills of a team and put that over a single players kill used for chart 4
            player_team = df2.loc[0,player_name]
            team_total_kills = grouped_df.loc[player_team, "total_kills"]
            print(team_total_kills)
            player_kill = df2.loc[2,player_name]
            player_kill = pd.to_numeric(player_kill)
            print(player_kill)
            st.write("h")
            data_parse = [player_kill,team_total_kills]
            st.write(data_parse)
            #chart_4_data = pd.DataFrame()
            #fig = px.pie(names = player_name,values = [player_kill,team_total_kills], hole= 0.5,title="fig" )
            #st.plotly_chart(fig)
            

print(data_parse)
            
            


            













chart_row_2 = st.container (key ="data2" , width = "stretch",height= 500)
with chart_row_2:
    col_1, col_2 = st.columns(2, gap="small", border=True)
    with col_1:
        st.title("Score over time" , text_alignment = "center")
        st.line_chart (data = df, x = "timestamp", y = ["red_score", "blue_score"],
            color =["#FF0000", "#0000FF"]
        )
    with col_2:
       st.line_chart (df, x = "timestamp", y = [f"{player_name}"] )
        
        



    


