#%% ***LIBRARIES***
import pandas as pd
from pandas.core.indexes.base import Index
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools
import numpy as np
import streamlit as st

st.set_page_config(layout='wide') #centered or wide
header_container = st.beta_container()
visualization_container = st.beta_container()

@st.cache(allow_output_mutation=True)
def get_data():
    data_c = pd.read_excel('Data.xlsx', sheet_name='MasterList', usecols=['AREA', 'TAG_ID', 'WEIGHT', 'DEL_DATE', 'ISS_DATE', 'ERE_DATE']) 
    data_c.rename(columns={'TAG_ID': 'ITEM'}, inplace=True)
    return data_c

# ***STREAMLIT BODY***
with header_container:
    st.header('DBNR FCC - PANCOR')
    st.subheader('Steel Structure Progress, General and by Areas')
    st.write(
        """
        - Purple = Delivered by Samsung to Laydown area
        - Orange = Issued from Laydown area to FCC site
        - Green = Erected'
        """)

with visualization_container:
    
    time_scale = st.selectbox('Select time scale:', options=['Day', 'Week', 'Month'], index=0)
    data = get_data()
    # ***DATA LOAD***
    dates_cols = ['DEL_DATE', 'ISS_DATE', 'ERE_DATE']
    time_scale_dict = {'Day': '1D', 'Week': '1W', 'Month':'1M'}
    #%% ***TIME SCALE***
    dates = pd.date_range(start=data[dates_cols].min().min(), end=data[dates_cols].max().max(), freq=time_scale_dict[time_scale])
    # ***DATA PROCESSING COUNT ITEMS***
    totales = data.groupby(['AREA']).agg({'ITEM':'size'})
    rowgeneral = totales.sum()
    rowgeneral.name = 'GENERAL'
    totales = totales.append(rowgeneral)
    totales = totales.iloc[np.arange(-1, len(totales)-1)].unstack().reset_index(level=0, drop=True)

    table_collection_ci = {}

    for date_col in dates_cols:
        table = pd.pivot_table(data, values=['ITEM'], index=[date_col], columns=['AREA'], aggfunc={'ITEM':'size'}, fill_value=0).resample('1D').sum().cumsum()
        table = table.reindex(dates).fillna(method='ffill').fillna(0)
        table.columns = [i[1] for i in table.columns.to_flat_index()]
        table['GENERAL'] = table[list(table.columns)].sum(axis=1)
        table = table/totales
        table.index.name = 'DATE'
        table_collection_ci[date_col] = table.reset_index(drop=False)

    # ***DATA PROCESSING SUM ITEMS WEIGHT***
    totales = data.groupby(['AREA']).agg({'WEIGHT':'sum'})
    rowgeneral = totales.sum()
    rowgeneral.name = 'GENERAL'
    totales = totales.append(rowgeneral)
    totales = totales.iloc[np.arange(-1, len(totales)-1)].unstack().reset_index(level=0, drop=True)

    table_collection_sw = {}

    for date_col in dates_cols:
        table = pd.pivot_table(data, values=['WEIGHT'], index=[date_col], columns=['AREA'], aggfunc={'WEIGHT':'sum'}, fill_value=0).resample('1D').sum().cumsum()
        table = table.reindex(dates).fillna(method='ffill').fillna(0)
        table.columns = [i[1] for i in table.columns.to_flat_index()]
        table['GENERAL'] = table[list(table.columns)].sum(axis=1)
        table = table/totales
        table.index.name = 'DATE'
        table_collection_sw[date_col] = table.reset_index(drop=False)

    # ***VISUALIZATION***
    colors = ['#a07aff', '#ffa07a', '#7affa0']
    areas = np.roll(list(sorted(table.columns)),1)
    subplot_titles_c1=[area + ' - count items' for area in areas]
    subplot_titles_c2=[area + ' - sum weights' for area in areas]
    subplot_titles = list(itertools.chain(*zip(subplot_titles_c1, subplot_titles_c2)))
    fig = make_subplots(rows=len(areas), cols=2, subplot_titles=subplot_titles, shared_xaxes=False)

    for idx, area in enumerate(areas, start=1):
        for idj, date_col in enumerate(dates_cols):
            df = table_collection_ci[date_col]
            fig.add_trace(go.Bar(x=df['DATE'], y=df[area], marker_color=colors[idj], showlegend=False), row=idx, col=1)
            fig.update_yaxes(showgrid=True, tickvals=[0, 0.25, 0.5, 0.75, 1], tickformat='.0%', row=idx, col=1)
            fig.update_xaxes(tickformat='%d-%b', row=idx, col=1)

    for idx, area in enumerate(areas, start=1):
        for idj, date_col in enumerate(dates_cols):
            df = table_collection_sw[date_col]
            fig.add_trace(go.Bar(x=df['DATE'], y=df[area], marker_color=colors[idj], showlegend=False), row=idx, col=2)
            fig.update_yaxes(showgrid=True, tickvals=[0, 0.25, 0.5, 0.75, 1], tickformat='.0%', row=idx, col=2)
            fig.update_xaxes(tickformat='%d-%b', row=idx, col=2)

    fig.update_yaxes(range=[0,1])
    fig.update_layout(title_text='FCC Structure Areas Progress', barmode='overlay', bargap=0.0, height=3000)
    st.plotly_chart(fig, use_container_width=True)
    
#%% ***DATA CHECK***
data['CHECK'] = (data['ISS_DATE']>=data['DEL_DATE'])&(data['ERE_DATE']>=data['ISS_DATE'])