import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px

st.set_page_config(layout="wide")


st.sidebar.title("Movies Analysis")

choice = st.sidebar.radio(
    "Select an option",
    ("Overall Analysis", "Actor-Wise Analysis", "Director-Wise Analysis", "Country-Wise Analysis",
     "Writer-Wise Analysis", "Actors Comparison", "Directors Comparison",
     "Countries Comparison", "Writers Comparison")
)

df = pd.read_csv("movies_cleaned_updated.csv")
df = preprocessor.preprocess(df)

uncleaned_df = pd.read_csv("uncleaned_movies.csv")
uncleaned_df = preprocessor.preprocess_uncleaned(uncleaned_df)


if choice == "Overall Analysis":
    tab1, tab2 = st.tabs(['Overall Analysis', 'Missing Values Analysis'])
    with tab1:
        helper.overall_analysis(df)
    with tab2:
        helper.missing_value_analysis(uncleaned_df)

elif choice == "Actor-Wise Analysis":
    st.sidebar.title("Select Actor")
    helper.analyse(df, 'stars', 'Actor')

elif choice == "Director-Wise Analysis":
    st.sidebar.title("Select Director")
    helper.analyse(df, "directors", "Director")

elif choice == "Writer-Wise Analysis":
    st.sidebar.title("Select Writer")
    helper.analyse(df, "writers", "Writer")

elif choice == "Country-Wise Analysis":
    st.sidebar.title("Select Country")
    helper.analyse_country(df)

elif choice == "Actors Comparison":
    st.sidebar.title("Compare Actors")
    helper.compare(df, "stars", "Actor")

elif choice == "Directors Comparison":
    st.sidebar.title("Compare Directors")
    helper.compare(df, "directors", "Director")

elif choice == "Writers Comparison":
    st.sidebar.title("Compare Writers")
    helper.compare(df, "writers", "Writer")

else:
    st.sidebar.title("Compare Countries")
    helper.compare_countries(df)
