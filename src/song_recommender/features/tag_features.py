from collections import Counter

def clean_tags(tag_list, valid_tags):

    return [t for t in tag_list if t in valid_tags]

def get_tag_clusters(tag_list, tag_cluster_map):

    return [
        tag_cluster_map[t]
        for t in tag_list
        if t in tag_cluster_map
    ]

def dominant_cluster(clusters):
    if not clusters:
        return None

    return Counter(clusters).most_common(1)[0][0]

def add_tag_cluster_features(df, valid_tags, tag_cluster_map):

    df['clean_tags'] = df['tag_list'].apply(
        lambda tags: clean_tags(tags, valid_tags)
    )

    df['tag_set'] = df['clean_tags'].apply(
        lambda tags: set(tags) if tags else set()
    )

    df['tag_clusters'] = df['clean_tags'].apply(
        lambda tags: get_tag_clusters(tags, tag_cluster_map)
    )

    df['cluster_set'] = df['tag_clusters'].apply(
        lambda clusters: set(clusters) if clusters else set()
    )

    df['dominant_cluster'] = df['tag_clusters'].apply(dominant_cluster)

    return df