from ClusterUtils import DBScan
from ClusterUtils import KMeans
from ClusterUtils import KernelKM
from ClusterUtils import InternalValidator
from ClusterUtils import ExternalValidator
from IPython import embed
from ClusterUtils import Spectral
import numpy as np
##part II
# 1, Lloyd's kmeans three_glob.csv


km = KMeans(init='random', n_clusters=3, csv_path='Datasets/three_globs.csv')
km.fit_from_csv()
km.show_plot()
km.save_plot()
km.save_csv()

km = KMeans(init='k-means++', n_clusters=3, csv_path='Datasets/three_globs.csv')
km.fit_from_csv()
km.show_plot()
km.save_plot()
km.save_csv()

km = KMeans(init='global', n_clusters=3, csv_path='Datasets/three_globs.csv')
km.fit_from_csv()
km.show_plot()
km.save_plot()
km.save_csv()
# 2, Hartigan
km = KMeans(algorithm = 'hartigans', n_clusters=3, csv_path='Datasets/three_globs.csv')
km.fit_from_csv()
km.show_plot()
km.save_plot()
km.save_csv()
# 3，internal well_seperated
km = KMeans(csv_path='Datasets/well_separated.csv')
dfs = []
cs = []
for i in range(2, 10):
    km.n_clusters = i # IMPORTANT -- Update the number of clusters to run.
    dfs.append(km.fit_predict_from_csv())
    cs.append(i)

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_cvnn_table()
iv.show_cvnn_plot()
#iv.save_cvnn_plot()

iv.make_silhouette_table()
iv.show_silhouette_plot()
iv.save_silhouette_plot()

iv.save_csv(cvnn=True, silhouette=True)

#4, image_segmentation.csv  external
km = KMeans(n_clusters=7, csv_path='Datasets/image_segmentation.csv')
km = KMeans(n_clusters=5, csv_path='Datasets/well_separated.csv')
km.fit_from_csv()
km.show_plot()
data = km.fit_predict_from_csv()
ev = ExternalValidator(data)
nmi = ev.normalized_mutual_info()
nri = ev.normalized_rand_index()
a = ev.accuracy()
print([nmi, nri, a])



# #5，ten times
# inter_si = []
exter_nmi = []
exter_nri = []
exter_a = []
km = KMeans(csv_path='Datasets/well_separated.csv', n_clusters=5,init='global')
for _ in range(10):
    data = km.fit_predict_from_csv()
    ev = ExternalValidator(data)
    nmi = ev.normalized_mutual_info()
    nri = ev.normalized_rand_index()
    a = ev.accuracy()
    exter_a.append(a)
    exter_nmi.append(nmi)
    exter_nri.append(nri)
    print([nmi, nri, a])
print(np.mean(np.array(exter_a)))
print(np.mean(np.array(exter_nmi)))
print(np.mean(np.array(exter_nri)))
print(np.std(np.array(exter_a)))
print(np.std(np.array(exter_nmi)))
print(np.std(np.array(exter_nri)))
dfs = []
cs = []
for _ in range(10):
    km.n_clusters = 5 # IMPORTANT -- Update the number of clusters to run.
    dfs.append(km.fit_predict_from_csv())
    cs.append(5)

iv = InternalValidator(dfs, cluster_nums=cs)
iv.make_silhouette_table()
si = iv.silhouette_table


 
# #III, DBScan
db = DBScan(eps=0.2, min_points=10, csv_path='Datasets/rockets.csv')
db.fit_from_csv()
db.show_plot()
# db.save_plot('DBScan plot')
# db.save_csv()
data = db.fit_predict_from_csv()
ev = ExternalValidator(data)
nmi = ev.normalized_mutual_info()
nri = ev.normalized_rand_index()
a = ev.accuracy()
print([nmi, nri, a])


kernel = KernelKM(n_clusters=2, csv_path='Datasets/eye_dense.csv')
kernel.fit_from_csv()
kernel.show_plot()
kernel.save_plot('kernel_plot')
kernel.save_csv()

spectral = Spectral(n_clusters=2,csv_path='Datasets/eye_dense.csv' )
spectral.fit_from_csv()
spectral.show_plot()
spectral.save_plot('spectral')
spectral.save_csv()