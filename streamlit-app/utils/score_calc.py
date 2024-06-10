from sklearn.neighbors import KDTree


def calculate_score(final_2d_embeddings, sources_df, query_point, radius=1.5):
    # Build the KDTree with the embeddings
    tree = KDTree(final_2d_embeddings)

    # Perform the radius query
    indices = tree.query_radius([query_point], r=radius)

    # Retrieve the labels for these points from sources_df
    labels_within_radius = sources_df.iloc[indices[0]]['data_source']

    # Define the scoring system
    label_scores = {
        'paper': -1,
        'article': 1,
        'reddit': -0.5
    }

    # Map the labels to scores and sum them
    scores = labels_within_radius.map(label_scores)
    final_score = scores.sum()

    return final_score


def add_scores(sources_df, final_2d_embeddings):
    # Calculate scores for all points
    scores = [
        calculate_score(final_2d_embeddings=final_2d_embeddings, sources_df=sources_df, query_point=point, radius=1.5)
        for point in final_2d_embeddings]

    # Add scores as a new column to the DataFrame
    sources_df['score'] = scores
