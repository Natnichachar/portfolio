import sqlite3
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64
from pathlib import Path
from st_ant_carousel import st_ant_carousel
from datetime import datetime

DB_path= "PizzaDatabase.db"
conn = sqlite3.connect(DB_path)
cursor=conn.cursor()

def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def query_df(sql:str):
    return pd.read_sql_query(sql, conn)

def img_to_data_uri(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{b64}"

#building dashboard
st.set_page_config(initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Hide sidebar navigation (pages list) */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
topLeft,topMiddle, topRight = st.columns(spec=[1,4,1])
with topLeft:
    if st.session_state.logged_in:
            st.markdown(f"Welcome, **{st.session_state.username}**")
with topRight: 
    if st.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.auth_mode = "StandingBy"
        st.switch_page("firstPage.py")

PizzaBox= load_lottie("json/PizzaBoxOrder.json")

tab1, tab2, tab3 = st.tabs(["Main", "Order", "Membership"])

with tab1:
    st.title("PIZZATERIA", text_alignment ="center")

    st_lottie( PizzaBox, loop = True, height = 450)
    #----------------------------------------------------------------------------
    st.subheader("Hot deals!")
    carousel = [
        {"content": f"<img src='{img_to_data_uri('images/deal1.png')}' style='width:100%;object-fit:contain;'/>"},
        {"content": f"<img src='{img_to_data_uri('images/deal2.png')}' style='width:100%;object-fit:contain;'/>"},
        {"content": f"<img src='{img_to_data_uri('images/deal3.png')}' style='width:100%;object-fit:contain;'/>"},
    ]

    carousel_style = {
        "background-color": "#9E3A2F",   # cream background
        "border": "2px solid #F5F1EA",   # brick red border
        "border-radius": "16px",
        "overflow": "hidden"
    }

    st_ant_carousel(carousel,carousel_style=carousel_style,autoplay=True, autoplaySpeed=3000, height = 500)

    #----------------------------------------------------------------------------
    col3, col4 = st.columns(2)
    with col3:
        df_topPick = pd.read_sql_query(
            "SELECT count(Flavor) AS amount,Flavor FROM orders GROUP BY Flavor ORDER By amount DESC LIMIT 1",
            conn)
        topPick = df_topPick["Flavor"][0]
        amount = df_topPick["amount"][0]

        with st.container(border=True):
            st.subheader(f"Top Pick This Month:")
            st.markdown(f"**{topPick}** **!**")
            st.markdown(f"This flavor is ordered **{amount}** times")

            if topPick == "Hawaiian":
                st.image("images/Hawaiian_medal.png")
            elif topPick == "Mushroom":
                st.image("images/Mushroom_medal.png")
            else:
                st.image("images/Pepperoni_medal.png")


    with col4:
        with st.container(border=True):
            st.subheader("Rating and Reviews")

            df_comments=pd.read_sql_query("SELECT comments.CommentId, comments.Context, comments.Rating, comments.CommentDate, customers.FirstName, customers.LastName From comments INNER JOIN customers ON customers.CustomerId = comments.CustomerId ORDER BY Rating DESC LIMIT 3;",
                                          conn)
            
            for _, row in df_comments.iterrows():
                full_name = f"{row['FirstName']} {row['LastName']}"
                rating = int(row["Rating"])

                dt = datetime.strptime(row["CommentDate"], "%Y-%m-%d %H:%M:%S")
                pretty_date = dt.strftime("%d %b %Y, %H:%M")

                st.markdown(f"""
                **{full_name}**  
                {'⭐' * rating}

                {row['Context']}

                <small><i>Date: {pretty_date}</i></small>

                ---
                """, unsafe_allow_html=True)

with tab2:
    st.subheader("Flavor")
    optionFlavor=st.selectbox(
        "What flavor are you craving today?",
        ("Hawaiian", "Mushroom", "Pepperoni"),index=None
    )

    optionSize= st.selectbox(
        "What size do you prefer?",
        ("Small", "Medium", "Large"),index=None)
    optionSideDish= st.selectbox(
        "Any snacks?",
        ("French Fries", "Mashed Potatoes", "-"),index=None)
    optionDrinks= st.selectbox(
        "Any drinks?",
        ("Cola", "Milk Tea", "-"),index=None)
    
    if optionFlavor==None or optionSize==None or optionSideDish==None or optionDrinks==None:
        st.warning("Please select all options")
        st.stop()


    st.subheader("Your Order Summary:")
    if optionFlavor=="Hawaiian":
        Hawaiian = load_lottie("json/HawaiianPizza.json")
        st_lottie( Hawaiian, loop = True, height = 450)
    elif optionFlavor=="Mushroom":
        Mushroom = load_lottie("json/MushroomPizza.json")
        st_lottie( Mushroom, loop = True, height = 450)
    else:
        Pepperoni = load_lottie("json/PepperoniPizza.json")
        st_lottie( Pepperoni, loop = True, height = 450)
    
    options=(optionFlavor,optionSize, optionSideDish, optionDrinks)
    optionstr = ",".join(f"'{i}'" for i in options)
    df_total = pd.read_sql_query(f"SELECT ItemName,Price FROM menu WHERE ItemName IN ({optionstr});", conn)
    st.dataframe(df_total, hide_index=True)
    total=df_total["Price"].sum()

    df_tier = pd.read_sql_query(f"SELECT Tier FROM customers WHERE UserName == '{st.session_state.username}';", conn)
    userTier=df_tier["Tier"][0]

    df_tierPromotion = pd.read_sql_query(f"SELECT DiscountPercentage, Promotion FROM tierPromotion WHERE Tier == '{userTier}';", conn)
    DiscountPercentage=df_tierPromotion["DiscountPercentage"][0]
    Promotion=df_tierPromotion["Promotion"][0]
    DiscountCalculation=1-(DiscountPercentage/100)
    afterDiscount= total*DiscountCalculation
    st.markdown(f"""
        Price: **{str(total)}** baht \n
        Membership Tier: **{userTier}**  {Promotion} \n
        Total Price = **{str(afterDiscount)}** baht \n
        """, unsafe_allow_html=True)
    
    df_oldTotalSpending = pd.read_sql_query(f"SELECT TotalSpending FROM customers WHERE UserName == '{st.session_state.username}';", conn)
    oldTotalSpending=df_oldTotalSpending["TotalSpending"][0]
    newTotalSpending=oldTotalSpending+afterDiscount

    if st.button("place order", type="primary"):
        PizzaChef=load_lottie("json/PizzaChef.json")
        st_lottie(PizzaChef, loop=True, height=300)
        st.write("Your order is in the kitchen!")
        cursor.execute("""
            UPDATE customers
            SET TotalSpending = ?
            WHERE UserName = ?
        """,(newTotalSpending, st.session_state.username))
        conn.commit()

        cursor.execute("""
            UPDATE customers
            SET Tier = CASE
            WHEN TotalSpending >=1000 THEN "Silver"
            WHEN TotalSpending >=2000 THEN "Gold"       
        """)
        conn.commit()
with tab3:
    df_tierM = pd.read_sql_query(f"SELECT Tier FROM customers WHERE UserName == '{st.session_state.username}';", conn)
    userTierM=df_tierM["Tier"][0]

    df_tierPromotionM = pd.read_sql_query(f"SELECT DiscountPercentage, Promotion FROM tierPromotion WHERE Tier == '{userTier}';", conn)
    DiscountPercentageM=df_tierPromotionM["DiscountPercentage"][0]
    PromotionMembership=df_tierPromotionM["Promotion"][0]
    df_TotalSpendingM = pd.read_sql_query(f"SELECT TotalSpending FROM customers WHERE UserName == '{st.session_state.username}';", conn)
    TotalSpendingM = df_TotalSpendingM["TotalSpending"][0]
    if Promotion =="":
        Promotion = "No Discount"
    
    if userTierM =="Bronze":
        st.image(img_to_data_uri("images/TierBronze.png"))
    elif userTierM =="Silver":
        st.image(img_to_data_uri("images/TierSilver.png"))
    elif userTierM =="Gold":
        st.image(img_to_data_uri("images/TierGold.png"))
    
    bottomLeft, bottomRight = st.columns(spec=[1,1])
    with bottomLeft:
        with st.container(border=True):
            st.markdown(f"""
                Your Membership Tier: **{userTier}** \n
                Total Spending: **{TotalSpendingM}** baht \n
                Promotion: **{Promotion}** \n
                """, unsafe_allow_html=True)
    with bottomRight:
        st.image(img_to_data_uri("images/TierInfo.png"))


