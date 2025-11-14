import pandas as pd
import streamlit as st
import altair as alt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_hendrix():
    df = pd.read_csv("hendrix.csv")
    df = df[df['year'] == df['year'].max()]
    df = df.dropna()
    df['tooltip_info'] = df['country']

    X = df.drop(columns=["Unnamed: 0", "tooltip_info", "ccode", "country", "year", "yhat", "yhat_2"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["x"] = X_pca[:, 0]
    df["y"] = X_pca[:, 1]
    return df, X_scaled

# NOT RUN -- file is saved
def run_tsne():
    df = pd.read_csv("anes_trust_no_missing.csv")
    X = df.drop(columns=["Unnamed: 0", "resp_id", "any_missing", 
                                   'gender', 'highest_edu', 'party_lean'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=2,
                metric='cosine',
                perplexity=10)
    X_tsne = tsne.fit_transform(X_scaled)
    df["x"] = X_tsne[:, 0]
    df["y"] = X_tsne[:, 1]
    df.to_csv("anes_trust_tsne.csv", index=False)

#run_tsne()

@st.cache_data
def load_anes():
    df = pd.read_csv("anes_trust_tsne.csv")
    df['tooltip_info'] = df['resp_id']

    X = df.drop(columns=["Unnamed: 0", "resp_id", "any_missing", "tooltip_info",
                                   'gender', 'highest_edu', 'party_lean',
                                   'x', 'y'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


import altair as alt
import pandas as pd
import numpy as np

def create_comparison_bar_chart(cluster_df, total_df, column, title, category_order=None, value_labels=None):
    """
    Create a side-by-side bar chart comparing cluster vs total sample
    
    Parameters:
    -----------
    cluster_df : DataFrame
        The filtered cluster data
    total_df : DataFrame
        The complete sample data
    column : str
        Column name to visualize
    title : str
        Chart title
    category_order : list, optional
        Ordered list of categories to display
    value_labels : dict, optional
        Mapping of values to labels (e.g., {1: 'Strongly approve', 2: 'Approve'})
    """
    # Calculate proportions for cluster
    cluster_counts = cluster_df[column].value_counts(normalize=True).reset_index()
    cluster_counts.columns = ['category', 'proportion']
    cluster_counts['group'] = 'Selected Cluster'
    
    # Calculate proportions for total sample
    total_counts = total_df[column].value_counts(normalize=True).reset_index()
    total_counts.columns = ['category', 'proportion']
    total_counts['group'] = 'Total Sample'
    
    # Combine
    combined = pd.concat([cluster_counts, total_counts])
    
    # Apply labels if provided
    if value_labels:
        combined['category'] = combined['category'].map(value_labels)
    
    # Convert to percentage
    combined['percentage'] = combined['proportion'] * 100
    
    # Create sort encoding for categories
    if category_order:
        sort_order = category_order
    else:
        sort_order = None
    
    # Create chart
    chart = alt.Chart(combined).mark_bar().encode(
        x=alt.X('group:N', title=None, axis=None),  # Remove x-axis labels
        y=alt.Y('percentage:Q', title='Percentage', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('group:N', 
                       scale=alt.Scale(domain=['Selected Cluster', 'Total Sample'],
                                      range=['#1f77b4', '#9EBAD2FF']),
                       legend=alt.Legend(title='Group')),
        column=alt.Column('category:N', 
                         title=None,
                         sort=sort_order,
                         header=alt.Header(
                             labelAngle=0,
                             labelAlign='center',
                             labelBaseline='bottom',
                             labelPadding=10,
                             titleOrient='bottom'
                         )),
        tooltip=[
            alt.Tooltip('category:N', title='Category'),
            alt.Tooltip('group:N', title='Group'),
            alt.Tooltip('percentage:Q', title='Percentage', format='.1f')
        ]
    ).properties(
        width=80,
        height=300,
        title=title
    )
    
    return chart


def create_gender_chart(cluster_df, total_df):
    """Create gender distribution comparison"""
    # Calculate % women
    cluster_pct_women = (cluster_df['gender'] == 'Female').mean() * 100
    total_pct_women = (total_df['gender'] == 'Female').mean() * 100
    
    data = pd.DataFrame({
        'group': ['Selected Cluster', 'Total Sample'],
        'percentage': [cluster_pct_women, total_pct_women]
    })
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('group:N', title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('percentage:Q', title='% Women', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('group:N', 
                       scale=alt.Scale(domain=['Selected Cluster', 'Total Sample'],
                                      range=['#1f77b4', "#9EBAD2FF"]),
                       legend=None),
        tooltip=[
            alt.Tooltip('group:N', title='Group'),
            alt.Tooltip('percentage:Q', title='% Women', format='.1f')
        ]
    ).properties(
        width=250,
        height=300,
        title='% Women'
    )
    
    return chart


def create_v_questions_chart(cluster_df, total_df, question_order=None):
    """
    Create a chart showing average responses for all V questions
    
    Parameters:
    -----------
    cluster_df : DataFrame
        The filtered cluster data
    total_df : DataFrame
        The complete sample data
    question_order : list, optional
        Ordered list of question column names to display
    """
    # Get all V question columns
    if question_order is None:
        v_columns = [col for col in cluster_df.columns if col.startswith('approve_') or 
                     col.startswith('trust_') or col.startswith('gov_') or 
                     col.startswith('corrupt_')]
    else:
        v_columns = question_order
    
    # Calculate means for each V question
    cluster_means = cluster_df[v_columns].mean()
    total_means = total_df[v_columns].mean()
    
    # Create comparison dataframe
    comparison_data = []
    for col in v_columns:
        # Clean up column name for display
        question_name = ' '.join(col.split("_")[1:-1]).title()
        comparison_data.append({
            'question': question_name,
            'question_raw': col,  # Keep original for sorting
            'group': 'Selected Cluster',
            'mean_response': cluster_means[col]
        })
        comparison_data.append({
            'question': question_name,
            'question_raw': col,
            'group': 'Total Sample',
            'mean_response': total_means[col]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create sort order based on original column order
    if question_order:
        sort_order = [col.replace('_', ' ').title() for col in question_order]
    else:
        sort_order = None
    
    # Create chart
    chart = alt.Chart(comparison_df).mark_bar().encode(
        x=alt.X('group:N', title=None, axis=None),  # Remove x-axis labels
        y=alt.Y('mean_response:Q', title='(Dis)Trust (lower=more trust)', 
                scale=alt.Scale(domain=[1, 5])),
        color=alt.Color('group:N', 
                       scale=alt.Scale(domain=['Selected Cluster', 'Total Sample'],
                                      range=['#1f77b4', '#9EBAD2FF']),
                       legend=alt.Legend(title='Group')),
        column=alt.Column('question:N', 
                         title=None,
                         sort=sort_order,
                         header=alt.Header(
                             labelAngle=0, 
                             labelAlign='right',
                             labelBaseline='middle',
                             labelPadding=1,
                             titleOrient='bottom'
                         )),
        tooltip=[
            alt.Tooltip('question:N', title='Question'),
            alt.Tooltip('group:N', title='Group'),
            alt.Tooltip('mean_response:Q', title='Avg Response', format='.2f')
        ]
    ).properties(
        width=80,
        height=300,
        title='Average Distrust By Question'
    ).configure_header(
        labelFontSize=10
    )
    
    return chart


# Example usage in Streamlit:
def display_cluster_analysis(cluster_df, total_df):
    """
    Display all comparison charts for a selected cluster
    
    Parameters:
    -----------
    cluster_df : DataFrame
        Filtered data for selected cluster
    total_df : DataFrame
        Complete sample data
    """
    
    # Define education order
    edu_order = ['<HS', 'HS', 'Some college', 'Bachelors', 'Advanced Degree']
    
    # Define party order (customize as needed)
    party_order = ['Strong Dem', 'Not strong Dem', 'Lean Dem', 'Independent', 
                   'Lean Rep', 'Not strong Rep', 'Strong Rep']
    
    # Define V questions order (customize as needed)
    v_order = [
        'trust_gov_dc_V241229',
        'trust_court_system_V241230',
        'trust_people_V241234',
        'trust_election_officials_V241315',
        'trust_judiciary_V242419',
        'trust_news_media_V241335',
        'trust_social_media_V242422',
        'trust_parties_V242421',
        'trust_scientists_V242420'
    ]
    
    # 1. Gender comparison
    gender_chart = create_gender_chart(cluster_df, total_df)
    st.altair_chart(gender_chart, use_container_width=False)
    
    # 2. Education distribution
    edu_chart = create_comparison_bar_chart(
        cluster_df, 
        total_df, 
        'highest_edu',
        'Educational Attainment Distribution',
        category_order=edu_order
    )
    st.altair_chart(edu_chart, use_container_width=False)
    
    # 3. Party ID distribution
    party_chart = create_comparison_bar_chart(
        cluster_df,
        total_df,
        'party_lean',
        'Party Identification Distribution',
        category_order=party_order
    )
    st.altair_chart(party_chart, use_container_width=False)
    
    # 4. V questions average responses
    v_chart = create_v_questions_chart(cluster_df, total_df, question_order=v_order)
    st.altair_chart(v_chart, use_container_width=False)

st.header("k-Means Clustering")
st.markdown("Below are two datasets options for k-means clustering")

# Set two tabs
tab1, tab2 = st.tabs(["State Capacity (Hendrix)", "Trust (ANES)"])


num_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)
cluster_selected = st.sidebar.selectbox("Select a cluster for viewing", range(1, num_clusters+1))

def fit_kmeans(X_scaled, num_clusters):
    # scale the data
   
    # fit the model
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X_scaled)
    return model


def prep_viz(df, tooltip_vars = ['tooltip_info']):
    # add the cluster labels to the dataframe
# plot the clusters in 2D using streamlit
    c = (
        alt.Chart(df)
        .mark_circle(size=200, opacity=0.6)
        .encode(x=alt.X("x", title="PC1"),
                 y=alt.Y("y", title="PC2"),
                 color="cluster",
                 tooltip=tooltip_vars + ["cluster"]
                 )
         .interactive()
         )
    return c

with tab1:
    st.markdown("Hendrix (2010) examines different definitions or measures of state capacity (≈ the state's ability to repress, redistribute, or exercise power).")
    st.markdown("Here, we're using Hendrix's data to identify whether states cluster together based on variables related to state capacity (e.g., tax revenue, military spending, level of democracy).")
    with st.expander("Details on Hendrix (2010)"):
        st.markdown("Hendrix reviews the existing literature on state capacity, specifically in the context of the civil war literature, and identifies several arguments about how to measure state capacity, along with a set of variables that the previous literature argues are potential indicators of state capacity.")
        st.markdown("""He then runs principle components analysis and argues that empirically, variables related to state capacity can be summarized along three axis:" \
- "rational legality": high quality bureaucracies, high revenue, technologically advanced, usually democracies
- "rentier-autocraticness": high revenues from primary commodity exports, high government income as percent of GDP, low levels of democracy
- "neopatrimoniality":  high income, high primary commodity exports, *low* levels of taxation, high military spending.      
""")
        st.markdown("We're using the same data as Hendrix, and also doing PCA, but with the added step of clustering the data, not just scaling it along principal components.")
    st.markdown("While we do k-means clustering in 16 dimensions, it's impossible to visualize that directly. Instead, we'll use PCA* to reduce the data to two dimensions for plotting, with k-means clusters (on the original data) shown as color.")
    
    df_hendrix, X_scaled_hendrix = load_hendrix()


    model = fit_kmeans(X_scaled_hendrix, num_clusters)

    df_hendrix["cluster"] = model.labels_
    # prepend "cluster" to the cluster values
    df_hendrix["cluster"] = "cluster " + (df_hendrix["cluster"] + 1).astype(str)


    c = prep_viz(df_hendrix)
    # Draw the figure
    st.altair_chart(c, use_container_width=True, theme="streamlit")
    st.markdown("Showing just data for 1999.") 

    # Print the countries in the selected cluster
    selected_df = df_hendrix[df_hendrix["cluster"] == f"cluster {cluster_selected}"]
    countries = ', '.join(selected_df["country"].unique().tolist())
    st.markdown(f"Cluster {cluster_selected}: **{countries}**")

    with st.expander("*Details on PCA"):
        st.markdown("Principal Component Analysis (PCA) is a technique for reducing the dimensions of data while still preserving as much information as possible. You can think about it as rotating the axes of the data to find the directions of greatest variance.")
    with st.expander("Show all data"):
        st.write(df_hendrix)


with tab2:
    st.markdown("Data from ANES 2024 with questions about *dis*trust in institutions.")
    st.markdown("The figure below shows 2D t-SNE, with k-means clusters (on the original data) shown as color.")

    with st.expander("Details about the ANES"):
        st.markdown("The American National Election Survey is a long-running nationally representative survey (after weighting) of Americans about political opinions and political parties. Here, we're using data from the 2024 survey wave. Note that some questions were asked before the election, and others asked after the election. Consult an American politics expert before drawing too-strong conclusions from this.")
        st.markdown("Note that the scale that ANES uses is \"backward\": **higher** values indicate **less** trust.")
        st.markdown("The cluster analysis below plots average demographics and trust variables by cluster. Note that age is a restricted variable (for privacy) and isn't available in the standard ANES download.")
        st.markdown("""Clustering is done only on the following variables from ANES:
- trust_gov_dc (V241229)
- trust_court_system (V241230)
- corrupt_officials (V241233)
- trust_people (V241234)
- trust_election_officials (V241315)
- trust_judiciary (V242419)
- trust_gov (V242418)
- trust_congress (V242417)
- trust_news_media (V241335)
- trust_social_media (V242422)
- trust_parties (V242421)
- trust_scientists (V242420)
""")
        st.info("To provide slightly better visualization of the high dimensional data, this visualization uses t-SNE rather than PCA. Ask for details if you're interested!")

    df_anes, X_scaled_anes = load_anes()
    model_anes = fit_kmeans(X_scaled_anes, num_clusters)

    df_anes["cluster"] = model_anes.labels_
    df_anes["cluster"] = "cluster " + (df_anes["cluster"] + 1).astype(str)

    c = prep_viz(df_anes, tooltip_vars=['gender', 'highest_edu', 'party_lean'])
    # Draw the figure
    st.altair_chart(c, use_container_width=True, theme="streamlit")

    # Print the countries in the selected cluster
    st.markdown(f"## Analysis for Cluster {cluster_selected}")
    cluster_df = df_anes[df_anes["cluster"] == f"cluster {cluster_selected}"]
    st.markdown(f"**{cluster_df.shape[0]} respondents are in this cluster, {cluster_df.shape[0] / df_anes.shape[0]:.1%} of the total.**")
    #countries = ', '.join(selected_df["country"].unique().tolist())
    #st.markdown(f"Cluster {cluster_selected}: **{countries}**"
    display_cluster_analysis(cluster_df, df_anes)


    with st.expander("Show all data"):
        st.write(df_anes)