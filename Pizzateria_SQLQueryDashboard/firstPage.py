import streamlit as st
import sqlite3
import pandas as pd

DB_path= "PizzaDatabase.db"
conn = sqlite3.connect(DB_path)
cursor=conn.cursor()

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

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'auth_mode' not in st.session_state:
    st.session_state.auth_mode = "StandingBy"
if 'username' not in st.session_state:
    st.session_state.username = None

def login(username:str,password:str)-> tuple[bool, str]:
    df_login=pd.read_sql_query("""
        SELECT CustomerId,UserName, Password FROM customers 
        WHERE UserName=? AND PASSWORD =?;""",
        conn, params=(username, password))
    if len(df_login)==1:
        st.session_state.logged_in=True
        st.session_state.customer_id=df_login["CustomerId"][0]
        st.session_state.username = username
        return True, "Logged in successfully"
    else:
        st.session_state.logged_in=False
        return False, "Invalid username or password"
    
def checkUsername(username):
    df_checkUsername = pd.read_sql_query("""
        SELECT Username FROM customers WHERE Username = ?; """,
        conn, params=(username,))
    if len(df_checkUsername)==1:
        return False, "This username is already taken"
    else:
        return True, "Valid username"
    
def signup(UserName, Password, FirstName, LastName, Birthday, City):
    check, message = checkUsername(UserName)
    if check:
        cursor.execute("""
                    INSERT INTO customers (UserName, Password, FirstName, LastName, Birthday, City)
                    VALUES (?,?,?,?,?,?);""",
                    (UserName, Password, FirstName, LastName, Birthday, City))
        conn.commit()
        return True
    else:
        st.error(message)
        return False

if not st.session_state.logged_in:
    if st.button("Log In"):
        st.session_state.auth_mode = "LogingIn"
    if st.button("Sign Up"):
        st.session_state.auth_mode = "SigningUp"

if st.session_state.auth_mode == "LogingIn":
    with st.form(key="LogInForm"):
        UserName=st.text_input("User Name", key= "UserName")
        Password=st.text_input("Password", type="password")
        submit = st.form_submit_button("Submit", use_container_width=True)
        if submit:
            if checkUsername(UserName):
                ok, message = login(UserName,Password)
                if ok:
                    st.success(message)
                    st.switch_page("pages/mainPage.py")
                else:
                    st.error(message)

if st.session_state.auth_mode == "SigningUp":
    with st.form(key="SignUpForm"):
        UserName=st.text_input("User Name")
        Password=st.text_input("Password")
        FirstName=st.text_input("First Name")
        LastName=st.text_input("Last Name")
        Birthday=st.date_input("Birthday", value=None)
        City=st.text_input("City")
        submit = st.form_submit_button("Submit", use_container_width=True)
        if submit:
            if signup(UserName, Password, FirstName, LastName, Birthday, City):
                ok, message = login(UserName,Password)
                if ok:
                    st.success(message)
                    st.switch_page("pages/mainPage.py")
                else:
                    st.error(message)
