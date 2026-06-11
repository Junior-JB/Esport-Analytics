
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path



st.set_page_config(layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_VALIDATION_DIR = PROJECT_ROOT / "debug_validation"
ASSETS_DIR = PROJECT_ROOT / "assets" / "images_dashboard"

Score_df = pd.read_csv(DEBUG_VALIDATION_DIR / "final_score.csv")

df = pd.read_csv(DEBUG_VALIDATION_DIR / "telemetry_timeseries.csv")

df2 = pd.read_csv(DEBUG_VALIDATION_DIR / "match_summary.csv")

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




Enemy_image = Image.open(ASSETS_DIR / "Enemy_player.png")
target_height =80
target_width = 80
Resize_enemy_image= Enemy_image.resize((target_width,target_width))

Ally_image = Image.open(ASSETS_DIR / "ally_player.png")
resize_ally_image = Ally_image.resize((target_width,target_height))










title_container = st.container(key="title", width="stretch", height=100,border= False)
with title_container:
    st.title("ESPORTS DASHBOARD", text_alignment="center")


filter_bar = st.sidebar
with filter_bar:
    st.title("Sidebar filters will go here.")


kpi_container = st.container(key="kpis", width="stretch",border=False )
with kpi_container:
    col_1, col_2,final_score_col, col_3, col_4, col_5= st.columns(6,border=True,gap="small")  
    with col_1:

        player_name=st.selectbox( label = "name",options = df2.columns[1:9])
    
    with col_2:
        kd = df2.loc[1, player_name]
        st.markdown(f"""
        <p style="font-size:26px; font-weight:bold; text-align:center;">
            KD<br>
            {kd}
        </p>
                    
        """, unsafe_allow_html= True)
    with final_score_col:
        Ally_score = Score_df.loc[0, "blue_final_score"]
        Enemy_score= Score_df.loc[0,"red_final_score"]  
        st.markdown(f""" 
        <div style="display:flex;flex-direction:column;justify-content:center;align-items:center">
            <div style="font-size:26px">
                <b>Final Score</b>
            </div>
            <div style="display:flex;font-size:35px;align-item:center;">
                <span style="color:blue">
                    <b>{Ally_score}</b>
                </span>
                <span style="margin:0 10px">
                    <b>VS</b>
                </span>
                <span style="color:red">
                    <b>{Enemy_score}</b>
                </span>
            </div>
        
        </div>





        """,unsafe_allow_html= True)
    
    
    with col_3:

        Grenade_usage = int(df2.loc[4, "all_match"])
        st.markdown(f"""
        <p style="font-size:26px; font-weight:bold; text-align:center;">
            Grenade Usage<br>
            {Grenade_usage}
        </p>
        
                    
        """,unsafe_allow_html=True)

    
    
    with col_4:
        tactical_usage = int(df2.loc[5, "all_match"])
        st.markdown(f"""
        <p style="font-size:26px; font-weight:bold; text-align:center;">
            Tactical Usage<br>
           {tactical_usage}
        </p>                    
                    
        """,unsafe_allow_html=True)

    
    with col_5:
        Field_upgrade = int(df2.loc[6, "all_match"])
        st.markdown(f"""
        <p style="font-size:26px; font-weight:bold; text-align:center;">
            Field Upgrade<br>
            {Field_upgrade}
        </p>

        """,unsafe_allow_html=True)

        



chart_row_1 = st.container(key = "data" , width = "stretch")
with chart_row_1:
    chart_1_col_1, chart_3_col_3, chart_4_col_4 = st.columns([1.0,01.2,0.8])
    with chart_1_col_1:
        chart_5_col_1, chart_5_col_2 = st.columns([1,1])
        with chart_5_col_1:
            with st.container(key="ally_row_1", border=True,height = 100):
                col_player_1_image, col_player_1_kd = st.columns([0.7,1])
                with col_player_1_image:
                    st.image(resize_ally_image )

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
                    st.image(resize_ally_image)
                with col_player_2_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[2]} </b><br> 
                        KD: {df2.iloc[1,2]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                    
            with st.container(key="ally_row_3", border=True,height=100):
                col_player_3_image, col_player_3_kd = st.columns([0.7,1])
                with col_player_3_image:
                    st.image(resize_ally_image)
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
                    st.image(resize_ally_image)
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
                    st.image(Resize_enemy_image)

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
                    st.image(Resize_enemy_image)
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
                    st.image(Resize_enemy_image)
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
                    st.image(Resize_enemy_image)
                with col_player_8_kd:
                    st.markdown(f"""
                        <div style="font=size:24;"
                        <b>{df2.columns[8]} </b><br> 
                        KD: {df2.iloc[1,8]}
                                
                        </div>                                  
                    """, unsafe_allow_html=True)
                
                
            
 
            

    with chart_3_col_3:
        with st.container(key= "chart_3",border=True):

            fig_2 = px.bar(per_hill_df,x="hill_number", y=["nades_delta","stuns_delta","specialties_delta"],title = "Utility Usage by Hill",barmode = "stack")
            fig_2.for_each_trace(
                lambda t: t.update(
                    name={
                        "nades_delta": "Grendades",
                        "stuns_delta": "Tacticals",
                        "specialties_delta": "Field Upgrade"
                    }[t.name]
                )
            )
            fig_2.update_layout(legend=dict(orientation ="h",yanchor ="top", y= -0.15, xanchor = "center", x=0.5), margin=dict(t=20,b=50),height =415,title_x = 0.4,xaxis_title ="Hill Number", yaxis_title="Utility Used", 
            title=dict(
                text="Utility Usage By Hill",
                x=0.5,
                y=0.98,
                font=dict(size=26)
                
                )
                
            )
            

            st.plotly_chart(fig_2, use_container_width=True)
            



    with chart_4_col_4:
        with st.container(key="chart_4",border=True,width="stretch",height = 450):
            #uses parsed name to get total kills of a team and put that over a single players kill used for chart 4
            player_team = df2.loc[0,player_name]
            team_total_kills = grouped_df.loc[player_team, "total_kills"]
            player_kill = df2.loc[2,player_name]
            player_kill = pd.to_numeric(player_kill)
            remaining_team_kills = (team_total_kills - player_kill)
            remaining_team_kills = pd.to_numeric(remaining_team_kills)

            
            data_parse = pd.DataFrame({
                "category":["player_kills","remaining_team_kills"],
                "Kills": [player_kill,remaining_team_kills]
                
            })

            fig = px.pie(data_parse,
                names = "category", values ="Kills", hole= 0.5, color_discrete_sequence=["#0000FF", "#FFEE00"])
            fig.update_layout(showlegend=False,
                              
                annotations=[
                    dict(

                        text=f"<b>total team kills</b><br>{team_total_kills}",
                        x=0.5,
                        y=0.5,
                        font_size=13,
                        showarrow=False
                    )
                ],
                
                title=dict(
                    text=f"<b>Percentage of Team Kills<b>",
                    x=0.22,
                    y=0.98,
                    font=dict(size=26)
                    

                ),height=415
                              
            )
            st.plotly_chart(fig, use_container_width= True)
            

            

chart_row_2 = st.container (key ="data2" , width = "stretch",height= 500,border = False)
with chart_row_2:
    col_1, col_2 = st.columns(2, gap="small", border=True)
    with col_1:
        st.markdown(f"""
        <p style="font-size:26px;text-align:center;">
            <b>Score Overtime</b>
        </p>
        """,unsafe_allow_html=True)



        st.line_chart (data = df, x = "timestamp", y = ["red_score", "blue_score"],
            color =["#FF0000", "#0000FF"], y_label="Score", x_label="Time (s)"
        )
    with col_2:
       st.markdown(f"""
        <p style="font-size:26px;text-align:center;">
            <b>Player Kills Overtime</b>
        </p>

        """, unsafe_allow_html= True)
       st.line_chart (df, x = "timestamp", y = [f"{player_name}"],x_label="Time (s)",)
        
        



    

