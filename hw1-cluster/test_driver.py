from ClusterUtils import DBScan
from ClusterUtils import KMeans
from ClusterUtils import KernelKM
from ClusterUtils import InternalValidator
from ClusterUtils import ExternalValidator
from IPython import embed
from ClusterUtils import Spectral
###part II
## 1, Lloyd's kmeans three_glob.csv


# km = KMeans(init='random', n_clusters=3, csv_path='Datasets/three_globs.csv')
# km.fit_from_csv()
# km.show_plot()
# km.save_plot()
# km.save_csv()

# km = KMeans(init='k-means++', n_clusters=3, csv_path='Datasets/three_globs.csv')
# km.fit_from_csv()
# km.show_plot()
# km.save_plot()
# km.save_csv()

# km = KMeans(init='global', n_clusters=3, csv_path='Datasets/three_globs.csv')
# km.fit_from_csv()
# km.show_plot()
# km.save_plot()
# km.save_csv()
#2, Hartgan
# km = KMeans(init='k-means++',algorithm = 'hartigans', n_clusters=3, csv_path='Datasets/three_globs.csv')
# km.fit_from_csv()
# km.show_plot()
# km.save_plot()
# km.save_csv()
#4, image_segmentation.csv  external
# km = KMeans(n_clusters=7, csv_path='Datasets/image_segmentation.csv')
# km.fit_from_csv()
# km.show_plot()
# data = km.fit_predict_from_csv()
# ev = ExternalValidator(data)
# nmi = ev.normalized_mutual_info()
# nri = ev.normalized_rand_index()
# a = ev.accuracy()
# print([nmi, nri, a])



km = KMeans(init='global', csv_path='Datasets/image_segmentation.csv')
dfs = []
cs = []
for i in range(2, 10):
    km.n_clusters = i # IMPORTANT -- Update the number of clusters to run.
    #km.fit_from_csv()
    #km.show_plot()
    dfs.append(km.fit_predict_from_csv())
    cs.append(i)

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_cvnn_table()
iv.show_cvnn_plot()
iv.save_cvnn_plot()

iv.make_silhouette_table()
iv.show_silhouette_plot()
iv.save_silhouette_plot()

#iv.save_csv(cvnn=True, silhouette=True)

# db = DBScan(eps=0.3, min_points=10, csv_path='Datasets/rockets.csv')
# data = db.fit_predict_from_csv()
# embed()
# ev = ExternalValidator(data)
# nmi = ev.normalized_mutual_info()
# nri = ev.normalized_rand_index()
# a = ev.accuracy()


# db = DBScan(eps=0.3, min_points=10, csv_path='Datasets/rockets.csv')
# db.fit_from_csv()
# db.show_plot()
# db.save_plot('DBScan plot')
# db.save_csv()

kernel = KernelKM(n_clusters=2, csv_path='Datasets/eye_dense.csv')
kernel.fit_from_csv()
kernel.show_plot()
kernel.save_plot('kernel_plot')
kernel.save_csv()

# spectral = Spectral(n_clusters=2,csv_path='Datasets/eye_dense.csv' )
# spectral.fit_from_csv()
# spectral.show_plot()
# spectral.save_plot('spectral')
# spectral.save_csv()