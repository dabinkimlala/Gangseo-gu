import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
#from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from tqdm import tqdm

# ---------------- 1. 역지오코딩 -----------------
# Nominatim 객체 생성
geo_local = Nominatim(user_agent="South Korea")

# 역지오코딩 함수 정의
def reverse_geo(row, max_retries=3):
    lat = row["Y좌표"]  # 위도
    lon = row["X좌표"]  # 경도
    retries = 0
    while retries < max_retries:
        try:
            location = geo_local.reverse(f"{lat}, {lon}", timeout=10)
            if location:
                address_list = location.address.replace(" ", "").split(',')
                address_list.reverse()
                return " ".join(address_list[2:-1]) + " " + address_list[-1]
            else:
                return "Unknown"
        except GeocoderTimedOut:
            retries += 1
            time.sleep(1)  # 타임아웃 발생 시 대기 후 재시도
    return "Timeout"

# 버스 정류장 데이터 로드
df_bus = pd.read_csv('C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/강서구/데이터/서울시 버스 정류소 좌표 데이터(2022.11.30).csv', encoding='utf-8')

# 데이터 나누기 (200개씩)
chunk_size = 200
chunks = [df_bus.iloc[i:i + chunk_size] for i in range(0, len(df_bus), chunk_size)]

# 각 chunk에 대해 역지오코딩 수행
results = []
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}...")
    chunk["주소"] = [reverse_geo(row) for _, row in tqdm(chunk.iterrows(), total=len(chunk))]
    results.append(chunk)
    time.sleep(2)  # 과도한 요청 방지를 위한 대기 시간

# 모든 chunk를 병합
final_df = pd.concat(results, ignore_index=True)

# 결과 저장
final_df.to_csv('final_bus_station_with_address.csv', index=False, encoding='utf-8-sig')

print("최종 결과가 'final_bus_station_with_address.csv' 파일로 저장되었습니다.")

# CSV 파일 읽기 (파일 이름을 'data.csv'로 가정)
df1 = pd.read_csv("C:/Users/rlaek/OneDrive/바탕 화면/final_bus_station_with_address.csv")

# 왼쪽 7번째까지의 글자가 "서울특별시 강서구"인 데이터 필터링
df1 = df1[df1["주소"].str[:9] == "서울특별시 강서구"]

# 주소 데이터 전처리
def get_addrs(x) :
  x1 = x.split(" ")
  return " ".join(x1[2:3])

# 주소 전처리
df1["지번주소"] = df1["주소"].apply(get_addrs)

df1.groupby("지번주소")["정류소명"].count()

# ---------------- 2. pca -----------------
park = pd.read_csv("C:/Users/rlaek/OneDrive/바탕 화면/포트폴리오/강서구/데이터/park(데이터 분석용) (2).csv", encoding='utf-8')
park = park[["공원 수", "공원 면적률", "일반 주차장", "장애인 주차장", "노인 인구수", "외부장애", "내부장애 ", "정신적 장애", "대중교통 수", "사회복지시설 수", "총인구", "대중교통 이용량", "인구밀도"]]

sc = StandardScaler()
X = sc.fit_transform(park)
pca = PCA(n_components=3)

# 데이터 변환
X_transformed = pca.fit_transform(X)

# 분산비율
pca.explained_variance_ratio_

pca_components = pca.components_

columns = ["공원수", "공원면적률", "일반주차장 수", "장애인주차장 수", "노인인구", "외부장애 인구", "내부장애 인구", "정신적장애 인구", "대중교통 수", "사회복지시설 수", "총인구", "대중교통 이용량", "인구밀도"]

# PCA 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(pca_components , columns=columns)

# 주성분 이름 설정
pca_df.rename(index={0: "PCA1", 1: "PCA2", 2: "PCA3"}, inplace=True)

# 주성분 1에서 가장 큰 기여도를 가진 변수들의 인덱스 출력
important_vars = np.argsort(np.abs(pca_components[0]))[::-1][:2]
print("PCA1에서 가장 큰 기여도를 가진 변수들의 인덱스:", important_vars)

# 주성분 2에서 가장 큰 기여도를 가진 변수들의 인덱스 및 변수 이름 출력
important_vars_pca2 = np.argsort(np.abs(pca_components[1]))[::-1][:2]
print("PCA2에서 가장 큰 기여도를 가진 변수들의 인덱스:", important_vars_pca2)

# 주성분 3에서 가장 큰 기여도를 가진 변수들의 인덱스 및 변수 이름 출력
important_vars_pca3 = np.argsort(np.abs(pca_components[2]))[::-1][:2]
print("PCA3에서 가장 큰 기여도를 가진 변수들의 인덱스:", important_vars_pca3)

# 변수 이름 출력
print("PCA1에서 중요한 변수 이름:", [columns[i] for i in important_vars])
print("PCA2에서 중요한 변수 이름:", [columns[i] for i in important_vars_pca2])
print("PCA3에서 중요한 변수 이름:", [columns[i] for i in important_vars_pca3])

# ---------------- 3. Hierarchical Clustering -----------------
# 1) 사회적 약자 특성 변수 군집화 
pca0 = park[["장애인 주차장", "노인 인구수", "외부장애", "내부장애 ", "정신적 장애"]]

#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca0)

data_x = pd.DataFrame(data_x)

Y = pdist(data_x)

row_dist = pd.DataFrame(squareform(pdist(data_x, metric = "euclidean")))

row_clusters = linkage(pdist(data_x, metric = "euclidean"), method = "ward")
row_clusters = linkage(data_x.values, metric = "euclidean", method = "ward")
pd.DataFrame(row_clusters, index = ["cluster %d"%(i+1) for i in range(row_clusters.shape[0])])

# 덴드로그램
row_dendr = sch.dendrogram(row_clusters)

plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

k_range = range(2,7)
for k in k_range :
  average_clustering = AgglomerativeClustering(n_clusters = k, linkage = "ward")
  average_cluster = average_clustering.fit_predict(data_x)
  score = silhouette_score(data_x, average_cluster)
  print(score)

# 2) 인구 및 교통 특성 변수 군집화 
pca1 = park[["대중교통 이용량", "인구밀도", "대중교통 수", "총인구"]]
pca1

#정규화

data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca1)

row_clusters = linkage(pdist(pca1, metric = "euclidean"), method = "ward")
row_clusters = linkage(pca1.values, metric = "euclidean", method = "ward")
pd.DataFrame(row_clusters, index = ["cluster %d"%(i+1) for i in range(row_clusters.shape[0])])

# 덴드로그램

row_dendr = sch.dendrogram(row_clusters)

plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

k_range = range(2,7)
for k in k_range :
  average_clustering = AgglomerativeClustering(n_clusters = k, linkage = "ward")
  average_cluster = average_clustering.fit_predict(pca1)
  score = silhouette_score(pca1, average_cluster)
  print(score)
  
# ---------------- 4. K-means -----------------
# 1) 사회적 약자 특성 변수 군집화
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca0)

data_x = pd.DataFrame(data_x)

ks = range(1,8)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data_x)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(4, 4))

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters = 3).fit(data_x)

park["cluster"] = kmeans.labels_
park

# 2) 인구 및 교통 특성 변수 군집화 
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca1)

data_x = pd.DataFrame(data_x)
data_x

ks = range(1,8)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data_x)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.figure(figsize=(4, 4))

plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

kmeans = KMeans(n_clusters = 3).fit(data_x)

park["cluster"] = kmeans.labels_
park

# ---------------- 5. K-medoids Clustering -----------------
# 1) 사회적 약자 특성 변수 군집화
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca0)

# K-medoids clustering with different cluster numbers
inertias = []
silhouette_scores = []
for n_clusters in range(2, 9):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(data_x)
    inertias.append(kmedoids.inertia_)

# Elbow method
plt.plot(range(2, 9), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# K-medoids 모델 생성 및 학습
kmedoids = KMedoids(n_clusters=7, random_state=42)
clusters = kmedoids.fit(data_x)

# cluster 값을 원본 데이터에 넣기
park["cluster"] = clusters.labels_
park

# 실루엣 계수 계산
silhouette_avg = silhouette_score(data_x, kmedoids.labels_)
print("클러스터링 결과의 실루엣 계수: {:.3f}".format(silhouette_avg))

# 2) 인구 및 교통 특성 변수 군집화 
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca1)

data_x = pd.DataFrame(data_x)
data_x


# K-medoids clustering with different cluster numbers
inertias = []
silhouette_scores = []
for n_clusters in range(2, 9):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    kmedoids.fit(data_x)
    inertias.append(kmedoids.inertia_)

# Elbow method
plt.plot(range(2, 9), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# K-medoids 모델 생성 및 학습
kmedoids = KMedoids(n_clusters=6, random_state=42)
clusters = kmedoids.fit(data_x)

# cluster 값을 원본 데이터에 넣기
park["cluster"] = clusters.labels_
park

# ---------------- 6. GMM -----------------
# 1) 사회적 약자 특성 변수 군집화
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca0)

# X: input data, n_components: the number of mixture components to fit
def gmm_bic(X, n_components):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(X)
    log_likelihood = gmm.score(X)
    n_features = X.shape[1]
    n_params = n_components * (n_features + 1) * n_features / 2 + n_components * n_features + n_components - 1
    bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
    return bic

c = []
for i in range(2,8) :
  a = gmm_bic(data_x, i)
  a = c.append(a)
  print(c)

columns = ["BIC"]
bic = pd.DataFrame(c, columns=columns)
bic.rename(index={0:"2개", 1:"3개", 2:"4개", 3:"5개", 4:"6개", 5:"7개", 6:"8개"})

# 군집개수 = 3
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type = "diag", random_state=0, n_init=10)
gmm.fit(data_x)
gmm_label = gmm.predict(data_x)
park['gmm_label'] = gmm_label
park

# 2) 인구 및 교통 특성 변수 군집화 
#정규화
data_scaler = MinMaxScaler()
data_x = data_scaler.fit_transform(pca1)

c = []
for i in range(2,8) :
  a = gmm_bic(data_x, i)
  a = c.append(a)
  print(c)

columns = ["BIC"]
bic = pd.DataFrame(c, columns=columns)
bic.rename(index={0:"2개", 1:"3개", 2:"4개", 3:"5개", 4:"6개", 5:"7개", 6:"8개"})

# 군집개수 = 3
gmm = GaussianMixture(n_components=3, covariance_type = "diag", random_state=0, n_init=10)
gmm.fit(data_x)
gmm_label = gmm.predict(data_x)
park['gmm_label'] = gmm_label
park

# ---------------- 7. EDA -----------------
labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [13475,5737,38117]
explode = (0.1,0.1,0.2)
colors = ["gray","gray","orange"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie


labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [3961,1240,7459]
explode = (0.1,0.1,0.2)
colors = ["gray","gray","orange"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie



labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [522,196,1822]
explode = (0.1,0.1,0.2)
colors = ["gray","gray","orange"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie


labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]
sizes = [3.41,7.32,16.4,1.36]
explode = (0.1,0.1,0.2,0.1)
colors = ["gray","gray","orange","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie


labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]
sizes = [15323,3493,23924,34035]
explode = (0.1,0.1,0.1,0.2)
colors = ["gray","gray","gray","mediumpurple"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3"]
sizes = [5.8,31.3,39.3,0.6]
explode = (0.1,0.2,0.1,0.1)
colors = ["gray","#C42241","gray","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [9510,28238,34035]
explode = (0.1,0.1,0.2)
colors = ["gray","gray","orange"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie


labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [4,10.3,1.36]
explode = (0.1,0.2,0.1)
colors = ["gray","#C42241","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5", "cluster_6"]
sizes = [27396,12831,10634,4828,4385,10919,3100]
explode = (0.2,0.1,0.1,0.1,0.1,0.1,0.1)
colors = ["turquoise","gray","gray","gray","gray","gray","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5", "cluster_6"]
sizes = [5874,3933,2202,1270,842,3661,645]
explode = (0.2,0.1,0.1,0.1,0.1,0.1,0.1)
colors = ["turquoise","gray","gray","gray","gray","gray","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5", "cluster_6"]
sizes = [1200,287,340,234,129,699,82]
explode = (0.2,0.1,0.1,0.1,0.1,0.1,0.1)
colors = ["turquoise","gray","gray","gray","gray","gray","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5"]
sizes = [9030,9407,18314,13708,7310,33293]
explode = (0.1,0.1,0.1,0.1,0.1,0.2)
colors = ["gray","gray","gray","gray","gray","hotpink"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4", "cluster_5"]
sizes = [2.99,2.78,3.16,11.9,3.97,2.8]
explode = (0.1,0.1,0.1,0.2,0.1,0.1)
colors = ["gray","gray","gray","limegreen","gray","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [13351,23924,34035]
explode = (0.1,0.1,0.2)
colors = ["gray","gray","orange"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

labels = ["cluster_0", "cluster_1", "cluster_2"]
sizes = [4.1,16.4,1.36]
explode = (0.1,0.2,0.1)
colors = ["gray","#C42241","gray"]
plt.pie(sizes, explode = explode, colors=colors, labels = labels, autopct = "%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.pie

