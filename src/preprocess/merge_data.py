import pandas as pd


def merge_user_items_attributes(data_path):
    """
    data_path : .tsv or .csv가 있는 경로

    Returns :
    df : user & item & rating이 모두 합쳐진 dataframe
    two csv files : items.csv = 640KB & user_ratings_items.csv = 607MB
    """
    dt = pd.read_csv(data_path + "/directors.tsv", sep="\t")
    dt["director"] = dt["director"].apply(lambda x: x.replace("nm", "")).astype(int)
    dt = dt.groupby(["item"])["director"].apply(list)
    gn = (
        pd.read_csv(data_path + "/genres.tsv", sep="\t")
        .groupby(["item"])["genre"]
        .apply(list)
    )
    title = pd.read_csv(data_path + "/titles.tsv", sep="\t")
    train_ratings = pd.read_csv(data_path + "/train_ratings.csv")
    writers = pd.read_csv(data_path + "/writers.tsv", sep="\t")
    writers["writer"] = (
        writers["writer"].apply(lambda x: x.replace("nm", "")).astype(int)
    )
    writers = writers.groupby(["item"])["writer"].apply(list)
    years = pd.read_csv(data_path + "/years.tsv", sep="\t")
    data = title.copy()

    for d in [years, gn, dt, writers]:
        data = data.merge(d, on="item", how="left")
    y_null_idx = data["year"].isnull()
    w_null_idx = data["writer"].isnull()
    d_null_idx = data["director"].isnull()

    data.loc[y_null_idx, "year"] = data.loc[y_null_idx, "title"].apply(
        lambda x: int(x[-5:-1])
    )
    data.loc[w_null_idx, "writer"] = data.loc[w_null_idx, "writer"].apply(
        lambda x: [-999]
    )
    data.loc[d_null_idx, "director"] = data.loc[d_null_idx, "director"].apply(
        lambda x: [-999]
    )
    data = data.set_index("item")
    data.sort_index(inplace=True)
    data.to_csv(data_path + "/items.csv")
    data = train_ratings.merge(data, on="item", how="left")
    data.to_csv(data_path + "/user_ratings_items.csv")

    return data
