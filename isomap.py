import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import pdist, squareform
from adjustText import adjust_text
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
def deal_data():

    # 读取data文件，指定属性
    data = pd.read_table('zoo.data', header=None,engine='python',
                         names=['animal name','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','color'],sep='[,]')
    # 生成csv文件
    data.to_csv('zoo.csv', index=False)
    return data

def clean_data(data):
    data = np.array(data)
    newData = data[:, 1:-1]
    label = data[:,-1]
    newData_with_name = np.c_[data[:,0],newData]
    return newData, label, newData_with_name
def  Normalize(data):
    print(data[1,13],data.shape)
    data[:,13] =  (data[:,13]-np.min(data[:,13]))/(np.max(data[:,13])-np.min(data[:,13]))
    print(data[:,13])
    return data
def feature_weights(newData,label):
    ACC = []
    for i in range(0,len(newData[1])):
        #index = [a for a in range(101)]
        X = np.array(pd.DataFrame([newData[:,i]])).T # single feature
        y = np.array(pd.DataFrame([label])).T
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
        clf = SVC(C=1.0, kernel='linear')
        clf.fit(X_train, y_train)
        clf.score(X_train, y_train)
        clf.score(X_test, y_test)
        ACC.append(clf.score(X_test, y_test))
        print("feature", i, "ACC:", clf.score(X_test, y_test))
    features_weights = [i /np.sum(ACC) for i in ACC]
    features_weights_matrix =  np.mat(np.mat(np.ones(101)).T*(features_weights))
    features_weights_matrix = np.array(features_weights_matrix)
    newData = pd.DataFrame( np.multiply(newData,features_weights_matrix) )

    return newData

def calculate_distance_matrix(X,Y):
    '''
    :param X:
    :param Y:
    :return:
    '''
    D=np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            #print(np.sqrt(np.sum(np.power((X[i]-Y[j]),2))))
            D[i,j] = np.sqrt(np.sum(np.power((X[i]-Y[j]),2)))
    return D

# def calculate_distance_matrix_2(D):
#     dist = pdist(D, 'euclidean')     # 获得距离矩阵
#     dist = squareform(dist)          # 转化为方阵
#     return dist

def get_k_maxtrix(D, k):
    '''
    构建近邻矩阵
    输入：样本集D，近邻参数k
    输出：近邻矩阵k_dist
    '''
    # dist = pdist(D, 'euclidean')     # 获得距离矩阵
    # dist = squareform(dist)          # 转化为方阵

    inf = float('inf')
    m = D.shape[0]
    k_dist = np.ones([m, m])*1000
    #construct the Adjacency matrix of graph G with k nearest neighbor of each point
    for i in range(m):
        topk = np.argpartition(D[i], k)[:k + 1]
        k_dist[i][topk] = D[i][topk]
    return k_dist

def Floyd(X):
    '''
    Floyd算法：
    输入: 距离矩阵X：(m,m)
        算法核心 path[i,j]:=min{path[i,k]+path[k,j],path[i,j]}
        时间复杂度：O（n^3)，空间复杂度：O(n^2)
    '''
    m= X.shape[0]
    for k in range(m):
        for i in range(m):
            for j in range(m):
                X[i][j] = min( X[i][j], X[i][k]+X[k][j] )
    return X


def calculate_similarity_matrix(D):
    identity_matrix = np.mat(np.ones(D.shape[0]))
    R = D * identity_matrix.T * identity_matrix / D.shape[0]
    C = identity_matrix.T * identity_matrix * D / D.shape[0]
    C = np.array(C)

    R_C = identity_matrix.T * identity_matrix * D * identity_matrix.T * identity_matrix / np.power(D.shape[0], 2)
    S = -1 / 2 * (D - R - C + R_C)

    # print(S-S.T)
    #B = np.linalg.eigvals(S)
    # C = np.linalg.cholesky(S)
    # print(B)
    return S

def compute_eigen_decomposition(S, k=2):
    Eigenvalues, Eigenvectors = np.linalg.eigh(S)
    Eigenvalues_sort = np.argsort(-Eigenvalues)  # big->small
    Eigenvalues = Eigenvalues[Eigenvalues_sort]
    Eigenvectors = Eigenvectors[:, Eigenvalues_sort]
    Eigenvalues_First_K_Diag = np.diag(Eigenvalues[0:k])
    Eigenvectors_First_K = Eigenvectors[:, 0:k]
    X = np.dot(np.sqrt(Eigenvalues_First_K_Diag), Eigenvectors_First_K.T).T
    reconstruct_error = 1 - np.sum(Eigenvalues_First_K_Diag)/np.sum(np.diag(Eigenvalues[:]))

    return X,reconstruct_error

def Isomap(D, p, k):
    '''
    Isomap
    输入：样本集D，近邻参数p，降维后的维度k
    输出：降维后的数据X
    '''

    # D=calculate_distance_matrix_2(newData)
    K_DIST = get_k_maxtrix(D, p)
    dist = Floyd(K_DIST)
    S = calculate_similarity_matrix(dist)
    X,reconstruct_error = compute_eigen_decomposition(S,k)
    # B = np.linalg.eigvals(K_DIST)
    # print(B)
    # C = np.linalg.cholesky(K_DIST)
    #print(reconstruct_error)
    return X,p,reconstruct_error


def draw_data(newData,p):
    X = newData
    y = data.iloc[:,-1]
    #feature_names = data.columns.values.tolist()
    idx=[[]for i in range(8)]
    new_X = np.c_[X,y].astype(np.float)
    for i in range(0,new_X.shape[0]):
        for j in range(len(set(y))):
            if new_X[i,-1] == j+1:
                idx[j].append(i)
    color_map={0:'r',1:'b',2:'y',3:'m',4:'c',5:'g',6:'k'}
    new_X_with_name = np.c_[data.iloc[:,0],new_X]
    new_texts = []
    for i in range(len(set(y))):
        plt.scatter([new_X[idx[i],0]], [new_X[idx[i],1]], color=color_map[i], label=i+1)
        for j in range(len(new_X_with_name[idx[i],0])):
            new_X_with_name = np.array(new_X_with_name)
            #new_X_with_name[idx[i], 0][j] = [eval(a) for a in (new_X_with_name[idx[i],0][j])]
            texts = plt.text(new_X_with_name[idx[i],1][j], new_X_with_name[idx[i],2][j], (new_X_with_name[idx[i],0][j]),fontsize=10,verticalalignment="top", horizontalalignment="right")
            new_texts.append(texts)
        adjust_text(new_texts,
                    only_move={'texts': 'x',},
                    # arrowprops=dict(arrowstyle="-",
                    #                 color='black',
                    #                 lw=0.5))
                    )

    plt.xlabel('new_feature_1')
    plt.ylabel('new_feature_2')
    plt.suptitle('Isomap p=         ')
    plt.title(p)
    plt.legend()
    plt.figure(figsize=(10,10))
    plt.show(adjust=True)

if __name__=='__main__':
    data = deal_data()
    newData, label, newData_with_name = clean_data(data)
    newData = Normalize(newData)
    newData = feature_weights(newData,label)
    newData = np.array(newData)
    D = calculate_distance_matrix(newData, newData)
    X, p,reconstruct_error = Isomap(D, p=5, k=2)

    print(reconstruct_error)
    draw_data(X,p)