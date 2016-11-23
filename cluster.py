from numpy import square, sqrt, setdiff1d

CLUSTER = "cluster"


def cluster(frame, n_clusters=5, n_iter=1000):
    cols = setdiff1d(scaled.columns.tolist(), [CLUSTER])  # get the columns well use without the cluster
    centroids = frame.sample(n_clusters)  # randomly pick clusters. There are other ways to pick.
    ii = 0
    while True:
        ii += 1
        print(ii)
        frame[CLUSTER] = _cluster_assignments(frame[cols], centroids[cols])
        old_centroids = centroids.copy(deep=True)
        centroids = scaled.groupby(CLUSTER).mean()
        if old_centroids[cols].equals(centroids[cols]) or ii > n_iter:
            break
    return frame[CLUSTER]


def _cluster_assignments(frame, centroids):
    return scaled.apply(lambda x: _find_closest(x, centroids), axis=1)  # assign each point to a cluster


def _find_closest(point, points):
    distances = points.apply(_dist_from(point), axis=1)
    return distances.reset_index().sort_values(by=0).iloc[0].name


def _dist_from(point):
    return lambda x: _dist(x, point)


def _dist(point_one, point_two):
    return sqrt(square((point_one - point_two)).sum())


scaled = random_frame / random_frame.max()