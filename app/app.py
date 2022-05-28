import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from func import *

import streamlit as st

from plotly import tools
import plotly.graph_objs as go

# <======================================== Set up ========================================>
# For loading model only once
if 'rfr_model' not in st.session_state:
    df = read_csv_file('https://raw.githubusercontent.com/ZaikaBohdan/ds_car_price_proj/main/data/clean_train.csv')
    # Feature Engineering
    fe_df = all_col_to_col_flg(df)
    fe_df = brand_by_mean_price(fe_df, df)
    # Features/target split
    X, y = xy_split(fe_df)
    # Model fitting
    rfr = RandomForestRegressor(
        max_features='sqrt', 
        n_estimators=50, 
        random_state=0
        )
    rfr.fit(X, y)

    st.session_state.known_df = df
    st.session_state.rfr_model = rfr




# <======================================== Interface ========================================>
st.set_page_config(
    page_title='Used car price prediction for the CarDekho website',
    page_icon='🚗',
    layout='wide'
    )

st.write("# price prediction for the Automobile website")

web_pages = [
    'About the app', 
    'Predict the price of one car', 
    'Predict prices for a file with cars',
    'Explore car prices'
    ]
curr_web_page = st.sidebar.selectbox('Navigaton', web_pages)




# >>>>>>>>>> 'About the app' <<<<<<<<<<
if curr_web_page == 'About the app':
    st.markdown("""
    ## About the app
    This application is a part of the data science project ["Building a car price prediction model for the Automobile website"]. Through the "Navigation" in the sidebar, you can choose one of three options:
    1. **'Predict the price of one car'**: Manually enter vehicle characteristics to evaluate its price;
    2. **'Predict prices for a file with cars'**: Upload csv file with the characteristics of cars to evaluate their prices. If the column *"selling_price_inr"* is present in the given file, then also *MAE, MSE* and *R^2* metrics will be calculated and shown.
    3. **'Explore car prices'**: Explore selling prices in collected data with the help of visualizations.
    """)




# >>>>>>>>>> 'Predict the price of one car' <<<<<<<<<<
if curr_web_page == 'Predict the price of one car':
    st.markdown("## Predict the price of one car")

    selectbox_vals = {
        'fuel': ['Petrol', 'Diesel', 'CNG', 'LPG'],
        'seller_type': ['Individual', 'Dealer'],
        'transmission': ['Manual', 'Automatic'],
        'owner': ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'],
        'seats': ['2', '4', '5', '6', '7', '8', '9', '10'],
        'brand': ['Hyundai', 'Mahindra', 'Chevrolet', 'Honda', 'Ford', 'Tata', 'Toyota', 'Maruti', 'BMW', 'Volkswagen', 'Audi', 'Nissan', 'Skoda', 'Mercedes-Benz', 'Datsun', 'Renault', 'Fiat', 'MG', 'Jeep', 'Volvo', 'Kia', 'Land Rover', 'Mitsubishi', 'Jaguar', 'Porsche', 'Mini Cooper', 'ISUZU']
    }
    
    c1, c2 = st.columns(2)

    brand = c1.selectbox('Brand', selectbox_vals['brand'])
    fuel = c1.selectbox('Fuel', selectbox_vals['fuel'])
    engine_cc = c1.number_input('Capacity of the engine (cc)', min_value=624.0, max_value=6752.0)
    max_power_bhp = c1.number_input('Engine power (bhp)', min_value=25.4, max_value=626.0)
    transmission = c1.radio('Transmission', selectbox_vals['transmission'])    

    seats = c2.selectbox('Number of seats', selectbox_vals['seats'])
    owner = c2.selectbox('Owner', selectbox_vals['owner'])
    year = c2.number_input('Year', min_value=1983, max_value=2022, value=2022)
    km_driven = c2.number_input('Kilometers driven', min_value=100, max_value=3800000)
    seller_type = c2.radio('Seller', selectbox_vals['seller_type'])

    all_vals = {
        'brand': brand,
        'fuel': fuel,
        'engine_cc': engine_cc, 
        'max_power_bhp': max_power_bhp, 
        'transmission': transmission,
        'seats': seats,
        'owner': owner,
        'year': year,
        'km_driven': km_driven,
        'seller_type': seller_type 
        }
    df = pd.DataFrame(all_vals, index=[0])

    col_but = st.columns(5)
    pred_button = col_but[2].button('Evaluate price')
    if pred_button:
        with st.spinner():
            result = data_prep_and_predict(
                df, 
                st.session_state.known_df, 
                st.session_state.rfr_model, 
                return_drop=False, 
                skip_dc=True
                )
        st.write(f'#### Evaluated price of the car: ₹ {result[0]:,.2f}') 




# >>>>>>>>>> 'Predict prices for a file with cars' <<<<<<<<<<
if curr_web_page == 'Predict prices for a file with cars':
    st.markdown("## Predict prices for a file with cars")

    with st.sidebar.header('Upload input file'):
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file with car characteristics",
            type=["csv"]
            )
        st.sidebar.markdown("""
            [Example of required file](https://github.com/ZaikaBohdan/ds_car_price_proj/blob/main/data/valid.csv)
            """)

    if uploaded_file is not None:
        st.markdown("### Input data")
        
        valid_df = read_csv_file(uploaded_file)
        st.dataframe(valid_df)
        
        st.markdown("### Predicted prices")

        with st.spinner():
            result = data_prep_and_predict(
                valid_df, 
                st.session_state.known_df, 
                st.session_state.rfr_model, 
                'selling_price_inr' in list(valid_df.columns)
                )
        
        pred_df = predict_df(valid_df, result[0])
        st.dataframe(pred_df)

        csv_file = convert_df(pred_df)
        st.download_button(
            label='📥 Download .csv file with predicted prices',
            data=csv_file, 
            file_name= 'predicted_car_prices.csv'
            )
        
        if len(result) == 3:
            st.markdown("#### Model scores")
            st.dataframe(result[2])

        if not result[1].empty:
            st.markdown("#### Dropped rows from input file")
            st.dataframe(result[1])
    
    else:
        st.info('Awaiting for csv file with car characteristics to be uploaded in the sidebar.')




# >>>>>>>>>> 'Explore car prices' <<<<<<<<<<
if curr_web_page == 'Explore car prices':
    df = st.session_state.known_df
    st.markdown("## Explore car prices")

    cols_dict = {
        'Brand': 'brand',
        'Fuel': 'fuel',
        'Capacity of the engine (cc)': 'engine_cc', 
        'Engine power (bhp)': 'max_power_bhp', 
        'Transmission': 'transmission',
        'Number of seats': 'seats',
        'Owner': 'owner',
        'Year': 'year',
        'Kilometers driven': 'km_driven',
        'Seller': 'seller_type' 
        }
    cols_list = sorted(list(cols_dict.keys()))
    
    select_col = st.sidebar.selectbox('Choose car characteristic for grouping prices', cols_list)
    col_gb = cols_dict[select_col]
    vals_col_gb = list(df[col_gb].unique())

    plots_flg = st.sidebar.radio('Display prices of groups in', ['one plot', 'separate plots'])

    price_range = st.sidebar.slider(
        'Select a range of values',
        25000, 
        3950000,
        (25000, 3950000)
        )
    
    if col_gb in ['engine_cc', 'max_power_bhp', 'km_driven']:
        n_bins = st.sidebar.slider(f'Number of bins for {select_col}', 2, 10, 5)

        df['bins'] = pd.cut(df[col_gb], n_bins).astype(str).str.strip(']').str.replace('(', 'from ').str.replace(',', ' to')
        vals_col_gb = list(df['bins'].unique())
        vals_col_gb.sort(key=lambda x: float(x.split(' to ')[0].strip('from ')))
        col_gb = 'bins'
        pass
    else:
        vals_col_gb = list(df[col_gb].unique())

    select_vals = st.sidebar.multiselect(
        'Select groups to include', 
        vals_col_gb,
        default = vals_col_gb[0]
        )

    mask = (df['selling_price_inr'] >= price_range[0]) & (df['selling_price_inr'] <= price_range[1])
    df = df[mask]

    hists = [
        go.Histogram(
            x=df.loc[df[col_gb] == val, 'selling_price_inr'], 
            name=str(val), 
            opacity=0.75
            ) for val in select_vals
    ]

    if plots_flg == 'one plot':
        layout = go.Layout(barmode='overlay')

        fig = go.Figure(data=hists, layout=layout)
        fig.update_xaxes(title_text='Selling Price (₹)', title_font={'size': 20})
        fig.update_yaxes(title_text='Frequency', title_font={'size': 20})

        
    
    else:
        fig = tools.make_subplots(rows=len(hists)//3+1, cols=3)

        i, j = 1, 1
        for hist in hists:
            fig.append_trace(hist, i, j)
            j += 1
            if j == 4:
                i += 1
                j = 1
        
    fig.update_layout(
        showlegend=True,
        title={
            'text': f'Histograms of Used Car Sale Price for different {select_col}',
            'y':.92,
            'x':.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30}
            }
        )
    st.plotly_chart(fig, use_container_width=True)