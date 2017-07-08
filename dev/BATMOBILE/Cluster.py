class Cluster:
    def __init__(self, cluster):
        self.cluster = cluster
        self.magnitude = compute_cluster_magnitude(cluster)
        self.position = compute_cluster_position(cluster)
