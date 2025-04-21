import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import preprocessor


def format_collections(number):
    if number >= 1000:
        return '$' + str(round(float(number / 1000), 2)) + ' Billion'
    else:
        return '$' + str(round(float(number), 2)) + ' Million'


def get_most_frequent_collaborators(selected_option, option_df, collaborator_col, collaborator_role):
    top_collaborators = option_df.explode(collaborator_col).groupby(collaborator_col).agg(
        total_movies = ('Title', 'count'),
        total_collections = ('grossWorldWide (in Millions)', 'sum')).sort_values(['total_movies', 'total_collections'], ascending = [False, False]).reset_index().head(5)
    top_collaborators['total_collections'] = top_collaborators['total_collections'].round(2)
    st.header(f'Top {collaborator_col.capitalize()} Who Frequently Worked with {selected_option} & Their Total Box Office Collections')
    fig = px.bar(
        top_collaborators,
        x = collaborator_col,
        y = 'total_movies',
        text = 'total_collections',
        labels = {collaborator_col : collaborator_role, 'total_movies' : 'Total Movies', 'total_collections' : 'Total Collections (in Millions)'}
        )
    st.plotly_chart(fig, key=f'top {collaborator_role} collaborations with {selected_option} chart')
    st.write("")


def get_most_frequent_collaborators_comparison(option_role, selected_option1, selected_option2, option1_df, option2_df,
                                    collaborator_col):
    top_collaborators1 = option1_df.explode(collaborator_col)[collaborator_col].value_counts().head(5).reset_index()
    top_collaborators1[option_role] = selected_option1
    top_collaborators2 = option2_df.explode(collaborator_col)[collaborator_col].value_counts().head(5).reset_index()
    top_collaborators2[option_role] = selected_option2

    top_collaborators = pd.concat([top_collaborators1, top_collaborators2])

    st.header(
        f"Most Frequent {collaborator_col.capitalize()} Collaborations of {selected_option1} and {selected_option2}")
    fig = px.bar(
        top_collaborators,
        x=collaborator_col,
        y='count',
        color=option_role,
        labels={collaborator_col: collaborator_col.capitalize(), 'count': 'Total Movies'}
    )
    st.plotly_chart(fig, key=f"top {collaborator_col.capitalize()} for {selected_option1} and {selected_option2} chart")
    st.write("")


def analyse(df, col, role):
    options = df.explode(col)[col].unique().tolist()
    options.sort()
    options.remove('Unknown')
    selected_option = st.sidebar.selectbox(f"Select {role}", options)
    st.title("Analysis of " + selected_option)
    exploded_df = df.explode(col)
    filtered_df = exploded_df[exploded_df[col] == selected_option]

    # Calculating Top Statistics
    total_movies = filtered_df.shape[0]
    total_wordwide = format_collections(filtered_df['grossWorldWide (in Millions)'].sum())
    total_us_canada = format_collections(filtered_df['gross_US_Canada (in Millions)'].sum())
    total_opening_weekend = format_collections(filtered_df['opening_weekend_Gross (in Millions)'].sum())
    rank = int(exploded_df.groupby(col)['grossWorldWide (in Millions)'].sum().rank(method='dense', ascending=False)[
                   selected_option])
    genre = filtered_df.explode("genres").groupby("genres").size().sort_values(ascending=False).index[0]

    # Showing top statistics
    st.header("Top Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Movies", value=total_movies)
    col2.metric(label="Rank", value=int(rank))
    col3.metric(label="Genre Specialization", value=genre)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Opening weekend Collections", value=total_opening_weekend)
    col2.metric(label="Worldwide Collections", value=total_wordwide)
    col3.metric(label="US & Canada Collections", value=total_us_canada)
    st.write("")

    # Year-wise total movies
    st.header(f'Year-wise Presence of {selected_option} Movies in the Dataset')
    movie_count_df = filtered_df.groupby("Year").size().reset_index(name='movie_count')
    all_years = range(2000, 2025)
    movie_count_df = movie_count_df.set_index('Year').reindex(all_years, fill_value=0).reset_index()

    # Create the line plot using Plotly
    fig = px.line(movie_count_df, x='Year', y='movie_count',
                  labels={'movie_count': 'Number of Movies', 'Year': 'Year'},
                  markers=True
                  )
    st.plotly_chart(fig, key="year wise movies chart")

    # Highest Grossing Movie
    st.header(f"Highest grossing movie of {selected_option} (Worldwide)")
    st.table(filtered_df[filtered_df['grossWorldWide (in Millions)'] == filtered_df['grossWorldWide (in Millions)'].max()])
    st.write("")

    # Highest Rated Movie
    st.header(f"Highest rated movie of {selected_option}")
    st.table(filtered_df[filtered_df['Rating'] == filtered_df['Rating'].max()])
    st.write("")

    # Year-wise audience engagement
    st.header("Year-wise audience engagement by " + selected_option)
    avg_vote_df = filtered_df.groupby("Year")['Votes'].mean().reset_index()
    avg_vote_df = avg_vote_df.set_index('Year').reindex(all_years, fill_value=0).reset_index()

    # Create the line plot using Plotly
    fig = px.line(avg_vote_df, x='Year', y='Votes',
                  labels={'Votes': 'Average Votes per Movie', 'Year': 'Year'},
                  markers=True
                  )
    st.plotly_chart(fig, key="year wise audience engagement chart")
    st.write("")

    # Year-wise total collections
    st.header("Year-wise gross worldwide collections by " + selected_option)
    collections_df = filtered_df.groupby("Year")['grossWorldWide (in Millions)'].sum().reset_index()
    collections_df = collections_df.set_index('Year').reindex(all_years, fill_value=0).reset_index()

    # Create the line plot using Plotly
    fig = px.line(collections_df, x='Year', y='grossWorldWide (in Millions)',
                  labels={'grossWorldWide (in Millions)': 'Total collections(in millions) per year', 'Year': 'Year'},
                  markers=True
                  )
    st.plotly_chart(fig, key="year wise collections chart")
    st.write("")

    # Most frequent genres
    st.header("Most Frequent genres done by " + selected_option)
    # Explode genres column
    genre_counts = filtered_df.explode("genres")["genres"].value_counts().head(5)
    # Convert to DataFrame for Plotly
    genre_df = genre_counts.reset_index()
    genre_df.columns = ["Genre", "Count"]
    fig = px.bar(genre_df, x="Genre", y="Count", text="Count")
    st.plotly_chart(fig, key="most frequent genres chart")
    st.write("")

    # Most frequent collaborators
    if role == 'Actor':
        get_most_frequent_collaborators(selected_option, filtered_df, "directors", "Director")
        get_most_frequent_collaborators(selected_option, filtered_df, "writers", "Writer")
    elif role == 'Director':
        get_most_frequent_collaborators(selected_option, filtered_df, "stars", "Actor")
        get_most_frequent_collaborators(selected_option, filtered_df, "writers", "Writer")
    else:
        get_most_frequent_collaborators(selected_option, filtered_df, "directors", "Director")
        get_most_frequent_collaborators(selected_option, filtered_df, "stars", "Actor")

    # Year-wise average rating analysis
    st.header('Yearly Trends in Average Movie Ratings')
    yearwise_rating = filtered_df.groupby("Year")['Rating'].mean().reset_index(name='Average Rating')
    yearwise_rating = yearwise_rating.set_index('Year').reindex(all_years, fill_value=0).reset_index()
    fig = px.line(
        yearwise_rating,
        x='Year',
        y='Average Rating',
        markers=True
    )
    st.plotly_chart(fig, key='yearly rating trends')
    st.write("")

    if total_movies > 3:
        st.header(f"Rolling Average (window = 3) of {selected_option}'s Movie Rating")
        filtered_df['Rolling Average Rating'] = filtered_df['Rating'].rolling(window=3).mean()
        fig = px.line(
            filtered_df,
            x='Year',
            y='Rolling Average Rating',
            markers=True
        )
        st.plotly_chart(fig, key='rolling average rating')


def analyse_country(df):
    exploded_df = df.explode("countries_origin")
    exploded_df.drop_duplicates(subset=['Title','Year','Duration (in Minutes)','MPA','Rating','Votes'],
                                keep='first', inplace=True)
    countries_list = exploded_df['countries_origin'].unique().tolist()
    countries_list.remove("Unknown")
    countries_list.sort()
    selected_country = st.sidebar.selectbox("Select Country", countries_list)
    st.title("Analysis of " + selected_country)
    country_df = exploded_df[exploded_df['countries_origin'] == selected_country]

    # Calculating Top Statistics
    total_movies = country_df.shape[0]
    worldwide_collections = format_collections(country_df['grossWorldWide (in Millions)'].sum())
    opening_weekend_collections = format_collections(country_df['opening_weekend_Gross (in Millions)'].sum())
    gross_us_canada = format_collections(country_df['gross_US_Canada (in Millions)'].sum())
    top_director = country_df.explode('directors').groupby('directors')['grossWorldWide (in Millions)'].sum().sort_values(
        ascending=False).index[0]
    top_actor = country_df.explode('stars').groupby('stars')['grossWorldWide (in Millions)'].sum().sort_values(
        ascending=False).index[0]
    st.header("Top Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Movies", value=total_movies)
    col2.metric(label="Top Director", value=top_director)
    col3.metric(label="Top Actor", value=top_actor)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Opening weekend Collections", value=opening_weekend_collections)
    col2.metric(label="Worldwide Collections", value=worldwide_collections)
    col3.metric(label="US & Canada Collections", value=gross_us_canada)
    st.write("")

    # Top production companies
    st.header("Top Production Companies by Total Movies Produced")
    company_counts = country_df.explode("production_companies")['production_companies'].value_counts()
    if 'Unknown' in company_counts.index:
        company_counts.drop(index="Unknown", inplace=True)
    company_counts = company_counts.head(5).reset_index()

    # Create a bar chart using Plotly
    fig = px.bar(company_counts, x="production_companies", y="count",
                 labels={'production_companies': 'Production Companies', 'count': 'Number of movies produced'},
                 text="count")
    st.plotly_chart(fig, key="top production companies chart")
    st.write("")

    # Top directors
    top_directors = country_df.explode('directors').groupby('directors').agg(
        gross_collection = ('grossWorldWide (in Millions)', 'sum'),
        average_collection = ('grossWorldWide (in Millions)', 'mean')
        ).sort_values(by = 'gross_collection', ascending = False).head(5).reset_index().rename(
            columns={'gross_collection': 'Gross Worldwide Collections', 'average_collection' : 'Average Collections '
                                                                                               'per Movie'})
    top_directors['Gross Worldwide Collections'] = (top_directors['Gross Worldwide Collections'].
                                                    apply(format_collections))
    top_directors['Average Collections per Movie'] = (top_directors['Average Collections per Movie'].
                                                    apply(format_collections))
    st.header("Top Directors")
    st.table(top_directors)
    st.write("")

    # Top Actors
    top_actors = country_df.explode('stars').groupby('stars').agg(
        gross_collection = ('grossWorldWide (in Millions)', 'sum'),
        average_collection = ('grossWorldWide (in Millions)', 'mean')
        ).sort_values(by = 'gross_collection', ascending = False).head(5).reset_index().rename(
            columns={'gross_collection': 'Gross Worldwide Collections', 'average_collection' : 'Average Collections '
                                                                                               'per Movie'})
    top_actors['Gross Worldwide Collections'] = (top_actors['Gross Worldwide Collections'].
                                                    apply(format_collections))
    top_actors['Average Collections per Movie'] = (top_actors['Average Collections per Movie'].
                                                    apply(format_collections))
    st.header("Top Actors")
    st.table(top_actors)
    st.write("")

    # Top Writers
    top_writers = country_df.explode('writers').groupby('writers').agg(
        gross_collection = ('grossWorldWide (in Millions)', 'sum'),
        average_collection = ('grossWorldWide (in Millions)', 'mean')
        ).sort_values(by = 'gross_collection', ascending = False).head(5).reset_index().rename(
            columns={'gross_collection': 'Gross Worldwide Collections', 'average_collection' : 'Average Collections '
                                                                                               'per Movie'})
    top_writers['Gross Worldwide Collections'] = (top_writers['Gross Worldwide Collections'].
                                                    apply(format_collections))
    top_writers['Average Collections per Movie'] = (top_writers['Average Collections per Movie'].
                                                    apply(format_collections))
    st.header("Top Writers")
    st.table(top_writers)
    st.write("")

    # Top filming locations
    st.header("Top Filming Locations")
    location_counts = country_df.explode("filming_locations")['filming_locations'].value_counts()
    if 'Unknown' in location_counts.index:
        location_counts.drop(index="Unknown", inplace=True)
    location_counts = location_counts.reset_index()
    location_counts = location_counts[(location_counts['filming_locations'] != 'Unknown') & (
                location_counts['filming_locations'] != selected_country)].head(5)

    fig = px.bar(location_counts, x="filming_locations", y="count",
                 labels={'filming_locations': 'Filming Locations', 'count': 'Number of movies shot'}, text="count")
    st.plotly_chart(fig, key="top filming locations chart")
    st.write("")

    # year-wise worldwide collections
    st.header("Year-wise worldwide collections")
    yearwise_collection_count = country_df.groupby("Year")['grossWorldWide (in Millions)'].sum().reset_index()
    all_years = range(2000, 2025)
    yearwise_collection_count = (yearwise_collection_count.set_index('Year').reindex(all_years, fill_value=0).
                                 reset_index())

    fig = px.line(yearwise_collection_count, x='Year', y='grossWorldWide (in Millions)',
                  labels={'grossWorldWide (in Millions)': 'Total worldwide collections(in Millions)', 'Year': 'Year'},
                  markers=True
                  )
    st.plotly_chart(fig, key="year wise worldwide collections chart")
    st.write("")

    # Highest Grossing Movie
    st.header(f"Highest Grossing Movie Produced in {selected_country}")
    highest_grossing_movie = country_df[
        country_df['grossWorldWide (in Millions)'] == country_df['grossWorldWide (in Millions)'].max()]
    st.table(highest_grossing_movie)
    st.write("")

    # Highest Rated Movie
    st.header(f"Highest Rated Movie Produced in {selected_country}")
    votes_filtered_df = country_df[country_df['Votes'] > country_df['Votes'].mean()]
    highest_rated_movie = votes_filtered_df[votes_filtered_df['Rating'] == votes_filtered_df['Rating'].max()]
    st.table(highest_rated_movie)
    st.write("")

    # Genre Distributions
    st.header('Genres Distribution')
    genre_count = country_df.explode("genres").groupby("genres")['genres'].value_counts().reset_index(name='count')
    fig_treemap = px.treemap(
        genre_count,
        path=['genres'],
        values='count'
    )
    st.plotly_chart(fig_treemap, key="genre distributions")
    st.write("")

    # Average rating analysis of popular genres
    st.header("Average Rating Analysis of Popular Genres")
    popular_genres = country_df.explode("genres")["genres"].value_counts().head(10).index.tolist()
    exploded_genres = country_df.explode("genres")
    genres_rating_df = exploded_genres[exploded_genres['genres'].isin(popular_genres)].groupby('genres')[
        'Rating'].mean().reset_index()
    genres_rating_df['Rating'] = genres_rating_df['Rating'].round(2)
    fig = px.bar(genres_rating_df, x='genres', y='Rating', text='Rating')
    st.plotly_chart(fig, key="rating analysis chart")

    # Average Ratings over the Years
    st.header('Yearly Trends in Average Movie Ratings')
    yearwise_rating = country_df.groupby("Year")['Rating'].mean().reset_index(name='Average Rating')
    yearwise_rating = (yearwise_rating.set_index('Year').reindex(all_years, fill_value=0).reset_index())
    fig = px.line(
        yearwise_rating,
        x='Year',
        y='Average Rating',
        markers=True
    )
    st.plotly_chart(fig, key='yearly avg ratings chart')
    st.write("")

    # Duration Distribution
    st.header("Duration Distribution")
    fig = px.box(
        country_df,
        x='Duration (in Minutes)',
        hover_data={'Title': True, 'Year': True, 'Rating': True}
    )
    st.plotly_chart(fig, key="duration distribution plot")
    st.write("")

    # MPA Rating Distributions
    st.header('MPA Rating Distribution')
    mpa_count = country_df[country_df['MPA'] != 'Unknown']['MPA'].value_counts().reset_index(name='count')
    fig_treemap = px.treemap(
        mpa_count,
        path=['MPA'],
        values='count'
    )
    st.plotly_chart(fig_treemap, key="mpa rating distribution chart")


def get_comparison_tables(exploded_df, option_df, col, selected_option):
    total_movies = option_df.shape[0]
    rank = int(exploded_df.groupby(col)['grossWorldWide (in Millions)'].sum().rank(method = 'dense', ascending = False)[selected_option])
    genre = option_df.explode("genres").groupby("genres").size().sort_values(ascending = False).index[0]
    worldwide_collections = option_df['grossWorldWide (in Millions)'].sum().round(2)
    us_canada_collections = option_df['gross_US_Canada (in Millions)'].sum().round(2)
    opening_weekend_collections = option_df['opening_weekend_Gross (in Millions)'].sum().round(2)
    avg_votes = option_df['Votes'].mean().round(2)
    df_table = pd.DataFrame({
        'Name': [selected_option],
        'Total Movies': [total_movies],
        'Rank': [rank],
        'Genre Specialization': [genre],
        'Worldwide Collections': [format_collections(worldwide_collections)],
        'US & Canada Collections': [format_collections(us_canada_collections)],
        'Opening Weekend Collections': [format_collections(opening_weekend_collections)],
        'Average Votes per Movie': [avg_votes]
    }).set_index('Name')
    df_chart = pd.DataFrame({
        'Name': [selected_option],
        'Worldwide Collections(Millions)': [worldwide_collections],
        'US & Canada Collections(Millions)': [us_canada_collections],
        'Opening Weekend Collections(Millions)': [opening_weekend_collections],
    })
    return df_table, df_chart


def compare(df, col, role):
    options_list = df.explode(col)[col].unique().tolist()
    options_list.sort()
    options_list.remove('Unknown')
    selected_option1 = st.sidebar.selectbox(f"Select {role}1", options_list)
    selected_option2 = st.sidebar.selectbox(f"Select {role}2", options_list)
    st.title(f"Comparison of {selected_option1} and {selected_option2}\n")
    exploded_df = df.explode(col)
    option1_df = exploded_df[exploded_df[col] == selected_option1]
    option2_df = exploded_df[exploded_df[col] == selected_option2]

    # getting comparison table and chart
    df_table1, df_chart1 = get_comparison_tables(exploded_df, option1_df, col, selected_option1)
    df_table2, df_chart2 = get_comparison_tables(exploded_df, option2_df, col, selected_option2)

    comparison_table = pd.concat([df_table1, df_table2])
    st.header("Comparison Table")
    st.table(comparison_table)
    st.write("")

    # lifetime collection comparison
    st.header("Box Office Collections Comparison")
    st.markdown(f"The bar chart shows the average (mean) box office collection, which highlights the overall financial "
                f"strength of movies from each {role}. However, the box plot provides a clearer picture by showing the "
                "median (typical earnings) and IQR (range of most movies), making it less affected by extreme "
                "blockbusters. Together, these visualizations give a balanced comparison of movie performances")
    st.markdown(f"If the mean box office collection is significantly higher than the median, it suggests that the "
                f"{role} has a few exceptionally high-grossing movies (outliers) that are inflating the average. "
                "On the other hand, if the mean and median are close, it indicates a more consistent box office "
                "performance across movies, with fewer extreme blockbusters skewing the data.")
    # comparison_chart = pd.concat([df_chart1, df_chart2])
    # # Melting the dataframe to make it compatible with a grouped bar chart
    # comparison_chart_melted = comparison_chart.melt(id_vars="Name", var_name="Metric", value_name="Value")
    #
    # fig = px.bar(
    #     comparison_chart_melted,
    #     x="Metric",
    #     y="Value",
    #     color="Name",
    #     barmode="group",
    #     labels={"Value": "Values (in Millions)", "Metric": "Metrics"},
    #     text="Value"
    # )
    # st.plotly_chart(fig, key="lifetime collections comparison chart")
    merged_df = pd.concat([option1_df, option2_df])

    # Comparing average collections
    avg_collection_metrics = merged_df.groupby(col)[[
        'grossWorldWide (in Millions)',
        'gross_US_Canada (in Millions)',
        'opening_weekend_Gross (in Millions)'
    ]].mean().reset_index()
    melted_metrics = avg_collection_metrics.melt(id_vars=col, var_name="Metric", value_name="Value")
    melted_metrics['Value'] = melted_metrics['Value'].round(2)

    fig = px.bar(
        melted_metrics,
        x='Metric',
        y='Value',
        color=col,
        text='Value',
        barmode='group',
        labels={"Value": "Values (in Millions)"},
        title="Average Box Office Collections Comparison"
    )

    st.plotly_chart(fig, key="avg_metrics_compare")

    # comparing usual collections
    fig = px.box(merged_df, y=col, x="grossWorldWide (in Millions)",
                 title="Usual Worldwide Box Office Collections (Median and IQR)",
                 labels={"grossWorldWide (in Millions)": "Worldwide Box Office Collection (in Millions)"},
                 points="outliers",
                 hover_data='Title'
                 )
    st.plotly_chart(fig, key="compare usual metrics")
    st.write("")

    # Comparing yearly gross collections
    st.header('Yearly Gross Worldwide Collections (in Millions)')
    option1_yearwise_collections = option1_df.groupby("Year")['grossWorldWide (in Millions)'].sum().reset_index()
    all_years = range(2000, 2025)
    option1_yearwise_collections = (option1_yearwise_collections.set_index('Year').reindex(all_years, fill_value=0).
                                    reset_index())
    option2_yearwise_collections = option2_df.groupby("Year")['grossWorldWide (in Millions)'].sum().reset_index()
    option2_yearwise_collections = (option2_yearwise_collections.set_index('Year').reindex(all_years, fill_value=0).
                                    reset_index())
    # Add a column to identify options in each DataFrame
    option1_yearwise_collections[role] = selected_option1
    option2_yearwise_collections[role] = selected_option2

    # Combine the two DataFrames
    combined_df = pd.concat([option1_yearwise_collections, option2_yearwise_collections])

    # Create the line plot
    fig = px.line(
        combined_df,
        x='Year',
        y='grossWorldWide (in Millions)',
        color=role,
        markers=True
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Gross Worldwide Collections (in Millions)'
    )

    st.plotly_chart(fig, key="yearly collections comparison chart")
    st.write("")

    # Comparing year-wise audience engagement
    st.header('Year-wise Audience Engagement')
    option1_yearwise_engagement = option1_df.groupby("Year")['Votes'].mean().astype("int").reset_index()
    option1_yearwise_engagement = (option1_yearwise_engagement.set_index('Year').reindex(all_years, fill_value=0).
                                    reset_index())
    option2_yearwise_engagement = option2_df.groupby("Year")['Votes'].mean().astype("int").reset_index()
    option2_yearwise_engagement = (option2_yearwise_engagement.set_index('Year').reindex(all_years, fill_value=0).
                                   reset_index())

    option1_yearwise_engagement[role] = selected_option1
    option2_yearwise_engagement[role] = selected_option2

    combined_df = pd.concat([option1_yearwise_engagement, option2_yearwise_engagement])
    fig = px.line(
        combined_df,
        x='Year',
        y='Votes',
        color=role,
        markers=True
    )
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Votes per Movie'
    )
    st.plotly_chart(fig, key="audience engagement comparison chart")
    st.write("")

    # Comparing Year-wise average ratings
    st.header('Year-wise Average Ratings')
    option1_avg_rating = option1_df.groupby('Year')['Rating'].mean().reset_index(name='Average Rating').set_index(
        'Year').reindex(all_years, fill_value=0).reset_index()
    option2_avg_rating = option2_df.groupby('Year')['Rating'].mean().reset_index(name='Average Rating').set_index(
        'Year').reindex(all_years, fill_value=0).reset_index()
    option2_avg_rating[role] = selected_option2
    option1_avg_rating[role] = selected_option1
    avg_rating_concat = pd.concat([option1_avg_rating, option2_avg_rating])
    fig = px.line(
        avg_rating_concat,
        x='Year',
        y='Average Rating',
        color=role,
        markers=True
    )
    st.plotly_chart(fig, key='yearly rating comparison')
    st.write("")

    # Comparing rolling average ratings
    if (option1_df.shape[0] > 3) and (option2_df.shape[0] > 3):
        option1_df['Rolling Average Rating'] = option1_df['Rating'].rolling(window=3).mean()
        option2_df['Rolling Average Rating'] = option2_df['Rating'].rolling(window=3).mean()
        combined_df = pd.concat([option1_df, option2_df])
        st.header("Rolling Average Ratings with Window size of 3")
        fig = px.line(
            combined_df,
            x='Year',
            y='Rolling Average Rating',
            color=col,
            markers=True
        )
        st.plotly_chart(fig, key='rolling rating average comparison chart')
        st.write("")

    # Comparing highest grossing movies
    st.header(f"Highest Grossing Movies of {selected_option1} and {selected_option2}")
    temp1 = option1_df[
        option1_df['grossWorldWide (in Millions)'] == option1_df['grossWorldWide (in Millions)'].max()].set_index(
        col).rename_axis(role)
    temp2 = option2_df[
        option2_df['grossWorldWide (in Millions)'] == option2_df['grossWorldWide (in Millions)'].max()].set_index(
        col).rename_axis(role)
    highest_grossing_movies = pd.concat([temp1, temp2])
    st.table(highest_grossing_movies)
    st.write("")

    # Comparing highest rated movies
    st.header(f"Highest Rated Movies of {selected_option1} and {selected_option2}")
    temp1 = option1_df[option1_df['Rating'] == option1_df['Rating'].max()].set_index(col).rename_axis(role)
    temp2 = option2_df[option2_df['Rating'] == option2_df['Rating'].max()].set_index(col).rename_axis(role)
    highest_rated_movies = pd.concat([temp1, temp2])
    st.table(highest_rated_movies)
    st.write("")

    # Comparing most frequent genres
    st.header(f'Most frequent genres of {selected_option1} and {selected_option2}')

    # Explode genres column
    genre_counts1 = option1_df.explode("genres")["genres"].value_counts().head(5)
    genre_df1 = genre_counts1.reset_index()
    genre_df1.columns = ["Genres", "Count"]
    genre_df1[role] = selected_option1

    genre_counts2 = option2_df.explode("genres")["genres"].value_counts().head(5)
    genre_df2 = genre_counts2.reset_index()
    genre_df2.columns = ["Genres", "Count"]
    genre_df2[role] = selected_option2

    # top_genres = pd.concat([top_rated_genres1, top_rated_genres2])
    most_frequent_genres = pd.concat([genre_df1, genre_df2])
    fig = px.bar(
        most_frequent_genres,
        x='Genres',
        y='Count',
        text = 'Count',
        color=role,
        barmode='group'
    )
    st.plotly_chart(fig, key="top genres comparison")
    st.write("")

    # Comparing most frequent collaborations
    if role == "Actor":
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "directors")
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "writers")
    elif role == "Director":
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "stars")
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "writers")
    else:
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "directors")
        get_most_frequent_collaborators_comparison(role, selected_option1, selected_option2, option1_df, option2_df,
                                        "stars")


def compare_countries(df):
    countries_exploded = df.explode("countries_origin")
    countries_exploded.drop_duplicates(subset=['Title','Year','Duration (in Minutes)','MPA','Rating','Votes'],
                                       keep='first', inplace=True)
    countries_list = countries_exploded['countries_origin'].unique().tolist()
    countries_list.remove("Unknown")
    countries_list.sort()

    # Taking Inputs
    selected_country1 = st.sidebar.selectbox("Select Country1", countries_list)
    selected_country2 = st.sidebar.selectbox("Select Country2", countries_list)

    # Filtering the dataframe for both selected countries
    country1_df = countries_exploded[countries_exploded['countries_origin'] == selected_country1]
    country2_df = countries_exploded[countries_exploded['countries_origin'] == selected_country2]

    # Getting comparison table and chart
    df_table1, df_chart1 = get_comparison_tables(countries_exploded, country1_df, 'countries_origin', selected_country1)
    df_table2, df_chart2 = get_comparison_tables(countries_exploded, country2_df, 'countries_origin', selected_country2)

    df_table1.rename(columns={'Genre Specialization': 'Most Genres Produced'}, inplace=True)
    df_table2.rename(columns={'Genre Specialization': 'Most Genres Produced'}, inplace=True)

    comparison_table = pd.concat([df_table1, df_table2])
    st.header("Overall Comparison")
    st.table(comparison_table)
    st.write("")
    comparison_chart = pd.concat([df_chart1, df_chart2])

    # Comparing both countries collections
    st.header("Box Office Collections Comparison")
    st.markdown("The bar chart shows the average (mean) box office collection, which highlights the overall financial "
                "strength of movies from each country. However, the box plot provides a clearer picture by showing the "
                "median (typical earnings) and IQR (range of most movies), making it less affected by extreme "
                "blockbusters. Together, these visualizations give a balanced comparison of movie performances across "
                "countries.")
    st.markdown("If the mean box office collection is significantly higher than the median, it suggests that the "
                "country has a few exceptionally high-grossing movies (outliers) that are inflating the average. "
                "On the other hand, if the mean and median are close, it indicates a more consistent box office "
                "performance across movies, with fewer extreme blockbusters skewing the data.")

    country1_df = countries_exploded[countries_exploded['countries_origin'] == selected_country1]
    country2_df = countries_exploded[countries_exploded['countries_origin'] == selected_country2]
    countries_df = pd.concat([country1_df, country2_df])[['countries_origin', 'grossWorldWide (in Millions)']]

    mean_df = countries_df.groupby("countries_origin").mean().reset_index()
    mean_df['grossWorldWide (in Millions)'] = mean_df['grossWorldWide (in Millions)'].round(2)
    fig = px.bar(mean_df, x="countries_origin", y="grossWorldWide (in Millions)",
                 title="Average (Mean) Box Office Collections by Country",
                 labels={"grossWorldWide (in Millions)": "Average Box Office Collection (in Millions)",
                         "countries_origin": "Country"},
                 text='grossWorldWide (in Millions)')  # Show values on bars

    fig.update_layout(template="plotly_white", yaxis_title="Box Office Collection (Millions)")
    st.plotly_chart(fig, key="country avg collections")

    fig = px.box(countries_df, y="countries_origin", x="grossWorldWide (in Millions)",
                 title="Typical (Median) Box Office Collection by Country",
                 labels={"grossWorldWide (in Millions)": "Box Office Collection (in Millions)",
                         "countries_origin": "Country"},
                 points="outliers",
                 log_x=True)  # Use log_x instead of log_y for horizontal orientation

    fig.update_traces(hovertemplate="Country: %{y}<br>Box Office: %{x}M")  # Show actual values on hover

    fig.update_layout(template="plotly_white", xaxis_title="Box Office Collection (Log Scale)")
    st.plotly_chart(fig, key="country collections distributions")
    st.write("")

    # Comparing genre distributions
    st.header("Genre Distribution Across Countries")
    genre_counts = pd.concat([country1_df, country2_df]).explode("genres").groupby(
        ['countries_origin', 'genres']).size().reset_index(name='count')
    fig_treemap = px.treemap(genre_counts,
                             path=['countries_origin', 'genres'],
                             values='count',
                             labels={'count': 'Number of Movies', 'countries_origin': 'Country', 'genres': 'Genres'})

    st.plotly_chart(fig_treemap, key="genre distribution comparison chart")
    st.write("")

    # year-wise average collections analysis
    st.header(f"Year-wise average collection comparison of {selected_country1} and {selected_country2}")
    df_combined = pd.concat([country1_df, country2_df])
    fig = px.line(
        df_combined.groupby(['countries_origin', 'Year'])['grossWorldWide (in Millions)'].mean().reset_index(),
        x='Year',
        y='grossWorldWide (in Millions)',
        color='countries_origin',
        labels={'countries_origin': 'Country', 'grossWorldWide (in Millions)': 'Average Collections per Movie'
                                                                               '(in Million)'},
        markers=True
    )
    st.plotly_chart(fig, key="year-wise collection comparison chart")
    st.write("")

    # Comparing top directors
    st.header(f"Top directors of {selected_country1} and {selected_country2}")
    country2_top_directors = country2_df.explode("directors").groupby("directors")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_directors = country1_df.explode("directors").groupby("directors")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_directors['Country'] = selected_country1
    country2_top_directors['Country'] = selected_country2
    top_directors = pd.concat([country1_top_directors, country2_top_directors])

    fig = px.bar(
        top_directors,
        x='directors',
        y='grossWorldWide (in Millions)',
        color='Country',
        labels={'directors': 'Directors', 'grossWorldWide (in Millions)': 'Worldwide Collections(Million)'},
        barmode='group'
    )
    st.plotly_chart(fig, key="directors comparison chart")
    st.write("")

    # Comparing top actors
    st.header(f"Top actors of {selected_country1} and {selected_country2}")
    country2_top_actors = country2_df.explode("stars").groupby("stars")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_actors = country1_df.explode("stars").groupby("stars")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_actors['Country'] = selected_country1
    country2_top_actors['Country'] = selected_country2
    top_actors = pd.concat([country1_top_actors, country2_top_actors])
    fig = px.bar(
        top_actors,
        x='stars',
        y='grossWorldWide (in Millions)',
        color='Country',
        labels={'stars': 'Actors', 'grossWorldWide (in Millions)': 'Worldwide Collections(Million)'},
        barmode='group'
    )
    st.plotly_chart(fig, key="actors comparison chart")
    st.write("")

    # Comparing top writers
    st.header(f"Top writers of {selected_country1} and {selected_country2}")
    country2_top_writers = country2_df.explode("writers").groupby("writers")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_writers = country1_df.explode("writers").groupby("writers")[
        'grossWorldWide (in Millions)'].sum().sort_values(ascending=False).head(5).reset_index()
    country1_top_writers['Country'] = selected_country1
    country2_top_writers['Country'] = selected_country2
    top_writers = pd.concat([country1_top_writers, country2_top_writers])
    fig = px.bar(
        top_writers,
        x='writers',
        y='grossWorldWide (in Millions)',
        color='Country',
        labels={'writers': 'Writers', 'grossWorldWide (in Millions)': 'Worldwide Collections(Million)'},
        barmode='group'
    )
    st.plotly_chart(fig, key="writers comparison chart")
    st.write("")

    # Comparing movie duration distributions
    # Selecting relevant columns
    st.header("Movie Duration Distribution Comaprison")
    df_filtered = pd.concat([country1_df, country2_df])[['countries_origin', 'Duration (in Minutes)']]

    fig = px.box(df_filtered,
                 x='Duration (in Minutes)',
                 y='countries_origin',
                 labels={'countries_origin': 'Country', 'Duration (in Minutes)': 'Duration (Minutes)'},
                 color='countries_origin')

    st.plotly_chart(fig, key="duration comparison chart")
    st.write("")

    # year-wise rating analysis
    st.header(f"Year-wise rating analysis of {selected_country1} and {selected_country2}")
    df_combined = pd.concat([country1_df, country2_df])
    fig = px.line(
        df_combined.groupby(['countries_origin', 'Year'])['Rating'].mean().reset_index(),
        x='Year',
        y='Rating',
        color='countries_origin',
        labels={'countries_origin': 'Country', 'Rating': 'Average Rating'},
        markers=True
    )
    st.plotly_chart(fig, key="rating analysis chart")

    # Finding Highest grossing movie for each country
    st.header(f"Highest Grossing Movies of {selected_country1} and {selected_country2}")
    temp1 = country1_df[
        country1_df['grossWorldWide (in Millions)'] == country1_df['grossWorldWide (in Millions)'].max()].set_index(
        "countries_origin").rename_axis("Countries")
    temp2 = country2_df[
        country2_df['grossWorldWide (in Millions)'] == country2_df['grossWorldWide (in Millions)'].max()].set_index(
        "countries_origin").rename_axis("Countries")
    highest_grossing_movies = pd.concat([temp1, temp2])
    st.table(highest_grossing_movies)
    st.write("")

    # Finding Highest Rated movies for each country
    st.header(f"Highest Rated Movies of {selected_country1} and {selected_country2}")
    country1_df_filtered = country1_df[country1_df['Votes'] >= country1_df['Votes'].mean()]
    country2_df_filtered = country2_df[country2_df['Votes'] >= country2_df['Votes'].mean()]
    temp1 = country1_df_filtered[country1_df_filtered['Rating'] == country1_df_filtered['Rating'].max()].set_index(
        "countries_origin").rename_axis("Countries")
    temp2 = country2_df_filtered[country2_df_filtered['Rating'] == country2_df_filtered['Rating'].max()].set_index(
        "countries_origin").rename_axis("Countries")
    highest_rated_movies = pd.concat([temp1, temp2])
    st.table(highest_rated_movies)
    st.write("")


def get_top_ten_entities(df, col, basis):
    exploded_df = df.explode(col)
    exploded_df = exploded_df[exploded_df[col] != 'Unknown']
    top_df = None
    label = None
    heading = None
    if basis == 'Rating':
        # global top_df, label, heading
        top_df = exploded_df.groupby(col).filter(lambda x: len(x) > 2).groupby(col).agg(
            basis_col=('Rating', 'mean'),
            total_movies=('Title', 'count')).sort_values(by='basis_col', ascending=False).reset_index().head(10)
        top_df['basis_col'] = top_df['basis_col'].round(2)
        label = 'Average Rating'
        heading = f"Top 10 {col.capitalize()} by Average Ratings per Movie"
    else:
        top_df = exploded_df.groupby(col).agg(
            basis_col=('grossWorldWide (in Millions)', 'sum'),
            total_movies=('Title', 'count')).sort_values(by='basis_col', ascending=False).reset_index().head(10)
        top_df['basis_col'] = (top_df['basis_col'] / 1000).round(2)
        label = 'Total Gross (in Billions)'
        heading = f"Top 10 {col.capitalize()} by Worldwide Gross Collection"

    fig = px.bar(
        top_df,
        x=col,
        y='basis_col',
        text='basis_col',
        labels={col: col.capitalize(), 'basis_col': label},
        title=heading
    )
    return fig


def missing_value_analysis(df):
    # Columns with missing values
    st.title("Missing Values Analysis")
    st.header("Columns with their Missing Values Percentage")

    null_df = df.isnull().mean() * 100
    null_df = null_df[null_df > 0].reset_index(name='Percentage of Missing Values').rename(
        columns={'index': 'Columns'})
    null_df['Percentage of Missing Values'] = null_df['Percentage of Missing Values'].round(2)
    st.table(null_df)
    st.write("")

    st.header("Analysis of Missing MPA Rating Values")
    # Year-wise missing mpa values
    st.subheader("Year-wise Missing MPA Ratings Analysis")
    yearwise_missing_mpa_pct = (df.groupby('Year').apply(lambda x: x.isnull().sum())['MPA'] / df[
        'MPA'].isnull().sum() * 100).reset_index(name='Missing Percentage')
    fig = px.line(
        yearwise_missing_mpa_pct,
        x='Year',
        y='Missing Percentage',
        markers=True
    )
    st.plotly_chart(fig, key="year-wise missing mpa chart")
    st.markdown("### Key Insights from the Chart")
    st.markdown("Initially from years 2000 to 2012, there has been a gradual decline in missing MPA rating values with "
                "lowest point in year 2012.")
    st.markdown("But, after year 2012, there has been a sharp increase in missing MPA rating values with highest point "
                "in year 2024.")
    st.write("")

    # Top 10 genres with missing mpa ratings
    st.subheader('Top 10 Genres with the Highest Percentage of Missing MPA Ratings')
    mpa_missing_pct = df.explode('genres').groupby('genres')['MPA'].apply(
        lambda x: x.isnull().mean() * 100).sort_values(ascending=False).head(10).reset_index(name='Missing Percentage')
    fig = px.bar(
        mpa_missing_pct,
        x='genres',
        y='Missing Percentage',
        labels={'genres': 'Genres'}
    )
    st.plotly_chart(fig, key="genres with missing mpa chart")
    st.markdown("### Key Insights from the Chart")
    st.markdown("Most of the genres which have missing MPA values are Low-Budget or Niche Categories")
    st.markdown("Some of these are - ")
    st.markdown("1) Documentary-related genres (like Food Documentary, Music Documentary, etc.)")
    st.markdown("2) Low-production-cost genres (Reality TV, News, Concert, etc.)")
    st.write("")

    # relation of missing mpa with collections
    st.subheader("Impact of Missing MPA Rating on Average Collections")
    mpa_analysis_df = df.groupby(df['MPA'].isnull())['grossWorldWide'].mean().reset_index().rename(
        columns={'MPA': 'MPA_Missing?', 'grossWorldWide': 'Average Worldwide Collections'})
    mpa_analysis_df['Average Worldwide Collections'] = mpa_analysis_df['Average Worldwide Collections'] / 1000000
    mpa_analysis_df['Average Worldwide Collections'] = mpa_analysis_df['Average Worldwide Collections'].apply(
        format_collections)
    st.table(mpa_analysis_df)
    st.markdown("### Key Insights from the Table")
    st.markdown("Movies with no MPA ratings have lesser performance in Box Office than movies with MPA ratings.")
    st.markdown("So, most of these are small and independent films.")
    st.write("")

    # Analysis for missing collection values
    st.header("Analysis of Missing Collection Values")
    temp = df[df['grossWorldWide'].isnull()]['gross_US_Canada'].isnull().mean()
    st.markdown(f"Percentage of movies with missing grossWorldWide also missing gross_US_Canada: {temp:.2%}")

    temp = df[df['opening_weekend_Gross'].isnull()]['gross_US_Canada'].isnull().mean()
    st.markdown(f"Percentage of movies with missing opening_weekend_Gross also missing gross_US_Canada: {temp:.2%}")

    temp = df[df['gross_US_Canada'].isnull()]['opening_weekend_Gross'].isnull().mean()
    st.markdown(f"Percentage of movies with missing gross_US_Canada also missing opening_weekend_Gross: {temp:.2%}")
    st.markdown("So, if a row has a missing collection value, then there is extremely high chance that it's other "
                "collection values are also missing.")
    st.write("")

    # top countries having missing collection values
    st.subheader("Top 5 countries which contain missing Gross Worldwide values")
    country_df = df.explode("countries_origin")
    country_df.drop_duplicates(subset=['Title', 'Year', 'Duration', 'MPA', 'Rating', 'Votes'],
                               keep='first', inplace=True)
    country_df = country_df[country_df['countries_origin'] != 'Unknown']
    missing_collection_df = country_df[country_df['grossWorldWide'].isnull()]
    missing_gross_pct = (missing_collection_df.groupby("countries_origin").size() / missing_collection_df.shape[
        0] * 100).sort_values(ascending=False).reset_index(name='missing_pct').head(5)
    fig = px.bar(
        missing_gross_pct,
        x='countries_origin',
        y='missing_pct',
        labels={'countries_origin': 'Country', 'missing_pct': '% of contribution'}
    )
    st.plotly_chart(fig, key="top countries with missing collection values chart")
    st.markdown("### Key Insights from the Chart")
    st.markdown("The above chart shows the top 5 countries sorted by their contribution to missing values in "
                "the dataset")
    st.markdown('For example, according to the above chart, of all the missing worldwide collections in the dataset, '
                '60% of them are from United States')
    st.markdown("United States is the biggest contributor to missing collection values. The main reason for it is that "
                "United States is the dominant country in our dataset.")
    st.write("")

    # top countries which have missing collection values in their own data
    st.subheader('Top 5 Countries with the Highest Percentage of Missing Worldwide Collection Data')
    countries_grouped = country_df.groupby("countries_origin").filter(lambda x: len(x) > 100).groupby(
        "countries_origin")
    missing_collection_pct = (countries_grouped.apply(lambda x: x['grossWorldWide'].isnull().mean(),
                                                      include_groups=False) * 100).sort_values(
        ascending=False).reset_index(name='Missing Percentage').head(5)
    fig = px.bar(
        missing_collection_pct,
        x='countries_origin',
        y='Missing Percentage',
        labels={'countries_origin': 'Country', 'Missing Percentage': '% of Missing Worldwide Collections'}
    )
    st.plotly_chart(fig, key="top countries with missing collection values in their own data")
    st.markdown("### Key Insights from the Chart")
    st.markdown("The above chart shows the top 5 countries which have highest percentage of worldwide collection "
                "values missing in their own country's movies dataset.")
    st.markdown("For example, according to the above chart, of all the movies produced in Australia, 10% of them have "
                "missing values.")
    st.markdown("From the above chart, we can see that Australia, Canada, Spain, Japan and United States are the top"
                " 5 countries who are not sharing their movies' collection details with Australia being the highest.")
    st.write("")

    # top 10 genres with missing collection values
    st.subheader("Top 10 Genres with the Highest Percentage of missing Gross Worldwide values")
    missing_gross_pct = (
                df.explode('genres').groupby('genres')['grossWorldWide'].apply(lambda x: x.isnull().mean()).sort_values(
                    ascending=False) * 100).reset_index(name='missing_pct')
    missing_gross_pct = missing_gross_pct[missing_gross_pct['genres'] != 'Unknown'].head(10)
    fig = px.bar(
        missing_gross_pct,
        x='genres',
        y='missing_pct',
        labels={'genres': 'Genre', 'missing_pct': '% of contribution'}
    )
    st.plotly_chart(fig, key="top countries having missing collection values chart")
    st.markdown("### Key Insight from the Chart")
    st.markdown("Genres which have highest percentage of missing collection values are Niche Genres like 'Holiday "
                "Animation', 'Cozy Mystery', 'Erotic Thriller', etc. which are less popular and less produced genres.")
    st.write("")

    # year-wise analysis of missing collection values
    st.subheader("Year-wise Analysis of Missing Worldwide Collections")
    missing_collection_df = df[df['grossWorldWide'].isnull()]
    missing_pct_by_year = (
                (missing_collection_df.groupby("Year").size() / df.groupby("Year").size()).fillna(0) * 100).reset_index(
        name='missing_pct')
    fig = px.line(
        missing_pct_by_year,
        x='Year',
        y='missing_pct',
        markers=True,
        labels={'missing_pct': 'Missing Percentage'}
    )
    st.plotly_chart(fig, key="year-wise missing collection values")
    st.markdown("### Key Insights from the Chart")
    st.markdown("Percentage of missing values were lower in years from 2001 to 2010 but high in years from 2000 to "
                "2001, 2011 to 2013 and 2020 to 2024.")
    st.markdown("There were no missing wordwide collection values from years 2014 to 2019.")
    st.markdown("Percentage of missing values were all time high in the year 2020 probably because of Covid.")
    st.write("")

    # Analysis of Missing Budget Values
    st.header("Analysis of Missing Budget Values")
    temp = (df['budget'].isnull().mean() * 100).round(2)
    st.markdown(f"Percentage of Missing Budget Values in the Dataset: {temp}%")
    st.markdown("So, more than half budget values are missing")

    # Yearly Analysis of Missing Budget Percentages
    st.subheader('Year-wise Analysis of Missing Budget Percentages')
    missing_budget_df = df[df['budget'].isnull()]
    budget_missing_df = (missing_budget_df.groupby("Year").size() / df.shape[0] * 100).sort_values(
        ascending=False).reset_index(name='Missing Percentage')
    fig = px.line(
        budget_missing_df,
        x='Year',
        y='Missing Percentage',
        markers=True
    )
    st.plotly_chart(fig, key="yearly analysis of missing budget values")
    st.markdown("### Key Insight from the Chart")
    st.markdown("There is no specific relation between year and missing budget values but year 2024 has most percentage"
                " of missing budget values.")
    st.write("")

    # Relation of Missing Budget with Collections
    st.subheader("Impact of Missing Budget Data on Average Collections")
    budget_collection_relation = df.groupby(df['budget'].isnull()).agg(
        grossWorldWide=('grossWorldWide', 'mean'),
        gross_US_Canada=('gross_US_Canada', 'mean'),
        opening_weekend_Gross=('opening_weekend_Gross', 'mean')
    ).reset_index()
    budget_collection_relation['grossWorldWide'] = (budget_collection_relation['grossWorldWide'] / 1000000).apply(
        format_collections)
    budget_collection_relation['gross_US_Canada'] = (budget_collection_relation['gross_US_Canada'] / 1000000).apply(
        format_collections)
    budget_collection_relation['opening_weekend_Gross'] = (
                budget_collection_relation['opening_weekend_Gross'] / 1000000).apply(format_collections)
    budget_collection_relation.rename(
        columns={'grossWorldWide': 'Avg. Worldwide Collections', 'gross_US_Canada': 'Avg. US & Canada Collections',
                 'opening_weekend_Gross': 'Avg. Opening Weekend Collections', 'budget' : 'Missing Budget?'},
        inplace=True)
    st.table(budget_collection_relation)
    st.markdown("### Key Insights from the Chart")
    st.markdown("Budget is missing for less successful movies.")
    st.markdown("Possible Reasons:-")
    st.markdown("1) Smaller/Independent Films May Not Report Budgets")
    st.markdown("2) If a movie flopped, studios may not see value in reporting the budget.")
    st.markdown("3) Bias in Data Source - IMDB may have incomplete budget data for smaller movies.")
    st.write("")

    # Top Genres with missing budget values
    st.subheader('Top 10 Genres with the Highest Percentage of Missing Budget Values')
    budget_missing_pct = (
                df.explode("genres").groupby("genres")['budget'].apply(lambda x: x.isnull().mean()) * 100).sort_values(
        ascending=False).reset_index(name='Missing Percentage')
    budget_missing_pct = budget_missing_pct[budget_missing_pct['genres'] != 'Unknown'].head(10)
    fig = px.bar(
        budget_missing_pct,
        x='genres',
        y='Missing Percentage',
        labels={'genres': 'Genres'}
    )
    st.plotly_chart(fig, key="genres with missing budget values missing")
    st.markdown("### Key Insights from the Chart")
    st.markdown("Most of the genres having high percentage of missing budget values are Low-Budget Categories:-")
    st.markdown("1) Documentary-related genres (like Food Documentary, Music Documentary, etc.)")
    st.markdown("2) Anime-related genres (like Shōnen, Seinen, etc.)")
    st.markdown("3) Low-production-cost genres (Reality TV, News, Concert, etc.).")
    st.write("")

    # Analysis of Missing Filming Location Values
    st.header("Analysis of Missing Filming Location Values")
    st.subheader("Impact of Missing Filming Location Data on Average Collections")
    location_collection_relation = df.groupby(df['filming_locations'] == 'Unknown').agg(
        grossWorldWide=('grossWorldWide', 'mean'),
        gross_US_Canada=('gross_US_Canada', 'mean'),
        opening_weekend_Gross=('opening_weekend_Gross', 'mean')
    ).reset_index()
    location_collection_relation['grossWorldWide'] = (location_collection_relation['grossWorldWide'] / 1000000).apply(
        format_collections)
    location_collection_relation['gross_US_Canada'] = (location_collection_relation['gross_US_Canada'] / 1000000).apply(
        format_collections)
    location_collection_relation['opening_weekend_Gross'] = (
                location_collection_relation['opening_weekend_Gross'] / 1000000).apply(format_collections)
    location_collection_relation.rename(
        columns={'grossWorldWide': 'Avg. Worldwide Collections', 'gross_US_Canada': 'Avg. US & Canada Collections',
                 'opening_weekend_Gross': 'Avg. Opening Weekend Collections', 'filming_locations' : 'Missing Filming '
                                                                                                    'Location?'}, inplace=True)
    st.table(location_collection_relation)
    st.markdown("### Key Insights from the Table")
    st.markdown("Lower collections suggest that movies with missing filming location data have smaller "
                "productions and limited releases.")
    st.write("")

    # Missing Filming Location Values by Genre
    st.subheader('Top 10 Genres with the Highest Percentage of Missing Filming Location Values')
    location_missing_pct = (df.explode("genres").groupby("genres")['filming_locations'].apply(
        lambda x: (x == 'Unknown').mean()) * 100).sort_values(ascending=False).reset_index(name='Missing Percentage')
    location_missing_pct = location_missing_pct[location_missing_pct['genres'] != 'Unknown'].head(10)
    fig = px.bar(
        location_missing_pct,
        x='genres',
        y='Missing Percentage',
        labels={'genres': 'Genres'}
    )
    st.plotly_chart(fig, key="Missing Filming Location Values by Genre")
    st.markdown("### Key Insights from the Chart")
    st.markdown("Missing filming_location is not random—it’s highly dependent on genre.")
    st.markdown("Genres with most missing values are primarily animation, documentaries, and TV formats, which don’t "
                "have clear filming locations.")
    st.write("")

    st.header("Overall Conclusion")
    st.markdown("1) Missing MPA rating values declined from 2000 to 2012 (lowest in 2012) but sharply increased after "
                "2012, peaking in 2024.")
    st.markdown("2) Low-budget and niche genres (e.g., documentaries, reality TV, holiday animation) consistently have "
                "missing values for MPA ratings, collections, budget, and filming locations.")
    st.markdown("3) The top 5 countries with the highest percentage of missing worldwide collection values for their "
                "own movies are Australia, Canada, Spain, Japan, and the United States, with Australia being the "
                "highest at 10%.")
    st.markdown("4) Missing gross collection values are strongly interdependent, with the U.S. as the largest "
                "contributor due to its dominance in the dataset.")
    st.markdown("5) Missing budget values are more common in smaller or independent films, especially in low-budget "
                "genres like documentaries and anime.")
    st.markdown("6) Missing filming location values are genre-specific, primarily affecting animation, documentaries, "
                "and TV formats, which often lack clear filming locations.")


def overall_analysis(df):
    st.title("Overall Analysis of the Dataset")
    st.write("")
    st.header("Dataset Overview")
    total_movies = df.shape[0]
    year_range = "2000-2024"
    total_genres = df.explode("genres")['genres'].nunique()
    total_directors = df.explode("directors")['directors'].nunique()
    total_actors = df.explode("stars")['stars'].nunique()
    total_countries = df.explode("countries_origin")['countries_origin'].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Movies", value=total_movies)
    col2.metric(label='Years-Range', value=year_range)
    col3.metric(label="Genres", value=total_genres)
    st.write("")

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Directors", value=total_directors)
    col2.metric(label="Actors", value=total_actors)
    col3.metric(label="Countries", value=total_countries)
    st.write("")

    # Dominant countries in the dataset
    st.header("Top 10 Most Dominant Countries in the Dataset")
    countries_exploded = df.explode("countries_origin")
    countries_exploded = countries_exploded[countries_exploded['countries_origin'] != 'Unknown']
    countries_exploded.drop_duplicates(subset=['Title', 'Year', 'Duration (in Minutes)', 'MPA', 'Rating', 'Votes'],
                                       keep='first', inplace=True)
    dominant_countries = countries_exploded.groupby("countries_origin").size().sort_values(ascending=False).head(
        10).reset_index(name='Total Movies')
    fig = px.bar(
        dominant_countries,
        x='Total Movies',
        y='countries_origin',
        text='Total Movies',
        labels={'countries_origin': 'Country'},
        orientation='h'
    )
    st.plotly_chart(fig, key="Dominant Countries Chart")
    st.markdown("### Key insights from the chart")
    st.markdown("The chart shows the top dominant countries in the dataset in which United States is most dominant "
                "followed by United Kingdom, France and India.")
    st.markdown("There is a huge dominance of United States indicating that the dataset is imbalanced country-wise.")
    st.write("")

    # Genre Distribution
    st.header('Genre Distribution')
    genre_count = df.explode("genres").groupby("genres").size().sort_values(ascending=False).reset_index(name="count")
    fig_treemap = px.treemap(
        genre_count,
        path=['genres'],
        values='count'
    )
    st.plotly_chart(fig_treemap, key="Genre Distribution Chart")
    st.markdown("### Key insights from the chart")
    st.markdown("The chart shows the genre distribution across the dataset.")
    st.markdown("'Drama' is the most dominant genre followed by Comedy, Romance, Thriller and Action which implies "
                "audience love comedic, romantic, thrilling and action movies the most.")
    st.markdown("Historical, Mockumentary, etc. are less produced genres implying these genres are not for all the "
                "audience as whole.")
    st.write("")

    # Top Performing Production Companies
    st.header("Top Production Companies by Worldwide Gross")
    company_df = df.explode("production_companies")
    company_stats = company_df.groupby("production_companies").agg(
        total_gross=('grossWorldWide (in Millions)', 'sum'),
        movie_count=('Title', 'count')
    ).sort_values(by='total_gross', ascending=False).head(10).reset_index()

    fig = px.bar(
        company_stats,
        x='total_gross',
        y='production_companies',
        text='movie_count',
        labels={"production_companies": "Production Company",
                "total_gross": "Total Worldwide Collections (in Millions)", "movie_count": "Total Movies"},
        orientation='h'
    )
    st.plotly_chart(fig, key="top production companies chart")
    st.write("")

    # Correlation Heatmap
    st.header("Correlation Heatmap")
    correlation_matrix = df.corr(numeric_only=True)
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        labels=dict(x='Metrics', y='Metrics', color='Correlation')
    )

    fig.update_layout(
        width=800,
        height=800,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, key="Heatmap")
    st.markdown("### Key insights from the Heatmap")
    st.markdown("There is a very high positive correlation between all the collection metrics.")
    st.markdown("Votes also have considerably high positive correlation with the collection metrics")
    st.markdown("Ratings show weak correlation with collection metrics, indicating that the highest-grossing movies "
                "aren’t always the most beloved by audiences. Their massive earnings often stem from factors beyond "
                "quality—such as being sequels to popular films with an established fanbase.")
    st.write("")

    # Top Directors by Worldwide Gross Collection and Average Rating per Movie
    st.header("Top 10 Directors by Worldwide Gross Collection and Average Ratings per Movie")
    fig1 = get_top_ten_entities(df, "directors", "Gross")
    fig2 = get_top_ten_entities(df, "directors", "Rating")
    tab1, tab2 = st.tabs(['Top 10 Directors by Worldwide Collections', 'Top 10 Directors by Average Ratings'])
    with tab1:
        st.plotly_chart(fig1, key='top 10 directors fig1')
    with tab2:
        st.plotly_chart(fig2, key='top 10 directors fig2')
    st.markdown("### Key insights from the above Charts")
    st.markdown("The above two charts show top 10 directors with two different aspects - ")
    st.markdown("1) Highest Worldwide Collections")
    st.markdown("2) Highest Average Ratings")
    st.markdown("Most of the directors are different in the above charts.")
    st.markdown("So, the above two charts clearly show that it is not guaranteed that directors producing highest "
                "grossing movies also make critically acclaimed movies.")
    st.write("")

    # Top Writers by Worldwide Gross Collection and Average Rating per Movie
    st.header("Top 10 Writers by Worldwide Gross Collection and Average Rating per Movie")
    fig1 = get_top_ten_entities(df, "writers", "Gross")
    fig2 = get_top_ten_entities(df, "writers", "Rating")
    tab1, tab2 = st.tabs(['Top 10 Writers by Worldwide Collections', 'Top 10 Writers by Average Ratings'])
    with tab1:
        st.plotly_chart(fig1, key='top 10 writers fig1')
    with tab2:
        st.plotly_chart(fig2, key='top 10 writers fig2')
    st.markdown("### Key insights from the above Charts")
    st.markdown("The above two charts show top 10 writers with two different aspects - ")
    st.markdown("1) Highest Worldwide Collections")
    st.markdown("2) Highest Average Ratings")
    st.markdown("Here also, we can see similar trend like we saw in the case of directors.")
    st.markdown("Most of the writers are different in the above charts.")
    st.markdown("So, the above two charts clearly show that it is not guaranteed that writers producing highest "
                "grossing movies also make critically acclaimed movies.")
    st.write("")

    # US & Canada collection vs rest of the world collection
    st.header("US & Canada Collections Vs Other Collections")
    fig = px.scatter(
        df,
        x='gross_US_Canada (in Millions)',
        y='outside_US_Canada_gross',
        labels={'outside_US_Canada_gross': 'Outside US & Canada Collection',
                'gross_US_Canada (in Millions)': 'US & Canada Collections'}
    )
    st.plotly_chart(fig, key="domestic vs international collection")
    st.markdown("### Key insights from the above Chart")
    st.markdown("Movies successful in the US & Canada often perform well internationally, but not always in a perfectly"
                " linear way.")
    st.markdown("Lower earners in US & Canada show more variation in international success as other factors like genre "
                "and cultural appeal also play a role in shaping worldwide collections.")
    st.write("")

    # Top 10 Highest Grossing Movies
    st.header("Highest-Grossing Movies Bubble Chart")
    top_movies = df.sort_values('grossWorldWide (in Millions)', ascending=False).head(10)
    top_movies['genres'] = top_movies['genres'].apply(lambda x: x[0])
    fig = px.scatter(
        top_movies,
        x="Year",
        y="grossWorldWide (in Millions)",
        size="grossWorldWide (in Millions)",
        color='genres',
        hover_name="Title",
        hover_data={"Year": True, "grossWorldWide (in Millions)": True},
        size_max=60,
        labels={'genres': 'Genres', "grossWorldWide (in Millions)": "Worldwide Collections (in Millions)"}
    )
    st.plotly_chart(fig, key="highest grossing movies chart")
    st.markdown("### Key insights from the above Chart")
    st.markdown("The above chart shows the top 10 all time highest grossing movies.")
    st.markdown("The size of the bubble is determined by the total number of votes received by the movie.")
    st.markdown("Almost all of them are larger than life movies.")
    st.markdown("So, we can clearly see that audience loves to see larger than life movies.")
    st.write("")

    # Top Actors
    st.header("Top 10 Actors by Worldwide Collections")
    fig = get_top_ten_entities(df, "stars", "Gross")
    st.plotly_chart(fig, key="top 10 actors chart")
    st.write("")

    # Most Frequent Writer Director Collaboration and Their Outcome
    st.header("Most Frequent Writer Director Collaboration and Their Outcome")
    writer_director_df = df.explode("directors").explode("writers")
    writer_director_df = writer_director_df[writer_director_df['directors'] != writer_director_df['writers']]
    writer_director_df = writer_director_df[
        (writer_director_df['directors'] != 'Unknown') & (writer_director_df['writers'] != 'Unknown')]
    top_collaborators = writer_director_df.groupby(['directors', 'writers']).agg(
        total_movies=('Title', 'count'),
        total_collection=('grossWorldWide (in Millions)', 'sum')
    ).sort_values(by=['total_movies', 'total_collection'], ascending=[False, False]).head(10).reset_index().rename(
        columns={'directors': 'Directors',
                 'writers': 'Writers', 'total_movies': 'Total Movies',
                 'total_collection': 'Total Collections (in Millions)'})
    st.table(top_collaborators)
    st.markdown("### Key insights from the above Table")
    st.markdown("The above table shows the most frequent writer-director collaborations.")
    st.markdown("The highest number of collaboration is 10 with director 'Ken Loach' and writer 'Paul Laverty'.")
    st.markdown("High number of collaborations indicate movies made by these pairs genrally perform better at "
                "box office and also loved by the audience.")
    st.write("")

    # Average Rating Comparison of Top 10 Dominant Countries
    st.header('Average Rating Comparison of Top 10 Dominant Countries')
    top_countries_rating = countries_exploded.groupby("countries_origin").agg(
        total_movies=('Title', 'count'),
        avg_rating=('Rating', 'mean')
    ).sort_values(by='total_movies', ascending=False).reset_index().head(10)
    fig = px.bar(
        top_countries_rating,
        x='avg_rating',
        y='countries_origin',
        text='total_movies',
        labels={'countries_origin': 'Country', 'total_movies': 'Total Movies', 'avg_rating': 'Average Rating'},
        orientation='h'
    )
    st.plotly_chart(fig, key="countries rating comparison chart")
    st.markdown("### Key insights from the above Chart")
    st.markdown("The above chart compares different countries in terms of quality of content they produce.")
    st.markdown("Japan and South Korea generally produce better quality of content than the other countries.")
    st.markdown("Australia, India and France also have comparatively good average ratings.")
    st.write("")

    # comparison of Monolingual and Multilingual movies
    st.header("Comparison of Monolingual and Multilingual Movies")
    tab1, tab2, tab3, tab4 = st.tabs(['By Total Movies', 'By Average Worldwide Collections', 'Average Ratings',
                                     'Average Votes'])
    with tab1:
        movies_frequency = df['Language Format'].value_counts().reset_index()
        fig = px.bar(
            movies_frequency,
            x='Language Format',
            y='count',
            labels={"count": "Total Movies"},
            title="Comparison of Monolingual and Multilingual Movies by Total Movies",
            text='count'
        )
        st.plotly_chart(fig, key="by total movies")
        st.markdown("### Key insights from the above chart")
        st.markdown("The chart shows the distribution of monolingual and multilingual movies in the dataset.")
        st.markdown("Clearly, monolingual movies dominate the dataset by having movies almost double than the "
                    "multilingual movies")
    with tab2:
        collections_df = df.groupby("Language Format")['grossWorldWide (in Millions)'].mean().reset_index()
        collections_df['grossWorldWide (in Millions)'] = collections_df['grossWorldWide (in Millions)'].round(2)
        fig = px.bar(
            collections_df,
            x='Language Format',
            y='grossWorldWide (in Millions)',
            labels={"grossWorldWide (in Millions)": "Average Worldwide Collections (in Millions)"},
            title="Comparison of Monolingual and Multilingual Movies by Total Movies",
            text='grossWorldWide (in Millions)'
        )
        st.plotly_chart(fig, key="by worldwide collections")
        st.markdown("### Key insights from the above chart")
        st.markdown("The chart compares the monolingual and multilingual movies by their average worldwide "
                    "collections per movie.")
        st.markdown("Multilingual movies tend to perform much better at box office than monolingual movies")
    with tab3:
        ratings_df = df.groupby("Language Format")['Rating'].mean().reset_index()
        ratings_df['Rating'] = ratings_df['Rating'].round(2)
        fig = px.bar(
            ratings_df,
            x='Language Format',
            y='Rating',
            title="Comparison of Monolingual and Multilingual Movies by Average Ratings",
            text='Rating'
        )
        st.plotly_chart(fig, key="by average ratings")
        st.markdown("### Key insights from the above chart")
        st.markdown("The chart compares the monolingual and multilingual movies by their average ratings per movie.")
        st.markdown("Both categories of movies have similar average ratings, with multilingual movies exhibiting a "
                    "slightly higher rating, suggesting a broader appeal among diverse audiences")
    with tab4:
        votes_df = df.groupby("Language Format")['Votes'].mean().reset_index()
        votes_df['Votes'] = votes_df['Votes'].round(2)
        fig = px.bar(
            votes_df,
            x='Language Format',
            y='Votes',
            title="Comparison of Monolingual and Multilingual Movies by Average Votes",
            text='Votes'
        )
        st.plotly_chart(fig, key="by average votes")
        st.markdown("### Key insights from the above Chart")
        st.markdown("The chart compares the monolingual and multilingual movies by their average votes per movie.")
        st.markdown("Multilingual movies have very high average votes than monolingual Movies, suggesting much larger "
                    "audience for multilingual movies")
    st.write("")

    # comparison of yearly trends of Multilingual and Monolingual movies
    st.header("Comparison of Yearly Trends of Multilingual and Monolingual Movies")
    tab1, tab2, tab3, tab4 = st.tabs(['By Total Movies', 'By Average Ratings', 'By Average Worldwide Collections',
                                      'By Average Votes'])
    yearly_analysis = df.groupby(["Year", "Language Format"]).agg(
        total_movies=('Title', 'count'),
        avg_rating=('Rating', 'mean'),
        avg_collections=('grossWorldWide (in Millions)', 'mean'),
        avg_votes=('Votes', 'mean')
    ).reset_index()
    cols = ['avg_rating', 'avg_collections', 'avg_votes']
    yearly_analysis[cols] = yearly_analysis[cols].round(2)
    with tab1:
        fig = px.line(
            yearly_analysis,
            x='Year',
            y='total_movies',
            color='Language Format',
            markers=True,
            labels={'total_movies': 'Total Movies'},
            title='Year-wise Total Movies Analysis of Monolingual and Multilingual Movies'
        )
        st.plotly_chart(fig, key="yearly total movies")
        st.markdown("### Key insights from the above Chart")
        st.markdown("Monolingual movies have consistently outnumbered multilingual movies.")
        st.markdown("Multilingual movie counts have remained relatively stable over the years.")
        st.markdown("Post-2020, monolingual movies have increased, while multilingual movies have declined. Probably "
                    "because of changing audience preferences or production challenges after COVID-19 pandemic")
    with tab2:
        fig = px.line(
            yearly_analysis,
            x='Year',
            y='avg_rating',
            color='Language Format',
            markers=True,
            labels={'avg_rating': 'Average Ratings'},
            title='Year-wise Rating Analysis of Monolingual and Multilingual Movies'
        )
        st.plotly_chart(fig, key="by average rating")
        st.markdown("### Key insights from the above Chart")
        st.markdown("Multilingual movies have consistently higher ratings than monolingual ones.")
        st.markdown("Monolingual movie ratings have improved over time.")
        st.markdown("Both categories saw a sharp dip in 2020 but recovered afterward.")
    with tab3:
        fig = px.line(
            yearly_analysis,
            x='Year',
            y='avg_collections',
            color='Language Format',
            markers=True,
            labels={'avg_collections': 'Average Worldwide Collections (in Millions)'},
            title='Year-wise Average Worldwide Collection Analysis of Monolingual and Multilingual Movies'
        )
        st.plotly_chart(fig, key="by yearly average collections")
        st.markdown("### Key insights from the above Chart")
        st.markdown("Multilingual movies consistently earn more worldwide than monolingual movies.")
        st.markdown("Multilingual movie collections show sharp fluctuations, especially around 2010 and 2020.")
        st.markdown("Monolingual movies have a steady but less globally expansive revenue trend.")
        st.markdown("Both categories show post-2020 recovery, likely due to cinema reopenings.")
    with tab4:
        fig = px.line(
            yearly_analysis,
            x='Year',
            y='avg_votes',
            color='Language Format',
            markers=True,
            labels={'avg_votes': 'Average Votes'},
            title='Year-wise Audience Engagement Analysis of Monolingual and Multilingual Movies'
        )
        st.plotly_chart(fig, key="by yearly audience engagement")
        st.markdown("### Key insights from the above Chart")
        st.markdown("Multilingual movies consistently receive higher audience engagement than monolingual movies.")
        st.markdown("A sharp drop in votes around 2020 suggests the impact of the COVID-19 pandemic.")
        st.markdown("By 2024-2025, engagement levels for both categories have nearly equalized.")



