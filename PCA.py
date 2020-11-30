import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from adjustText import adjust_text
def deal_data():

    # 读取data文件，指定属性
    data = pd.read_table('zoo.data', header=None,engine='python',
                         names=['animal name','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','color'],sep='[,]')
    # 生成csv文件
    data.to_csv('zoo.csv', index=False)
    return data
def data_centering(data):
    identity_matrix = np.mat(np.ones(16))
    print(identity_matrix)
    data=np.mat(data)
    data[:,1:-1] = data[:,1:-1]- (data[:,1:-1]*identity_matrix.T*identity_matrix)/101
    return data
def  Normalize(data):
    print(data[1,13],data.shape)
    data[:,13] =  (data[:,13]-np.min(data[:,13]))/(np.max(data[:,13])-np.min(data[:,13]))
    print(data[:,13])
    return data
def PCA_method(data):
    '''
    :param data: raw data(18 features)
    input features for pca: 16(remove animal_name and color);
    :return:    newData : 101*2
    '''
    #Before PCA method, the original varivace of each axis
    pca = PCA(n_components=16)
    pca.fit(data[:,1:-1])
    print('original variance ratio',pca.explained_variance_ratio_)
    print('original variance', pca.explained_variance_)
    sum_variance = pca.explained_variance_
    #After PCA method, the varivace of k largest axis
    pca = PCA(n_components=2)
    pca.fit(data[:, 1:-1])
    print('after pca variance ratio', pca.explained_variance_ratio_)
    print('after pca variance', pca.explained_variance_)
    reconstruct_error = 1 - np.sum(pca.explained_variance_[:])/np.sum(sum_variance[:])
    print(reconstruct_error)
    newData = pca.fit_transform(data[:,1:-1])
    newData.astype(np.int)
    return newData

def draw_data(newData,data):
    X = newData
    data=pd.DataFrame(data)
    y = data.iloc[:,-1]
    idx=[[]for i in range(8)]#record the class type of data point
    new_X = np.c_[X,y].astype(np.float)
    for i in range(0,new_X.shape[0]):
        for j in range(len(set(y))):
            if new_X[i,-1] == j+1:
                idx[j].append(i)
    color_map={0:'r',1:'b',2:'y',3:'m',4:'c',5:'g',6:'k'}
    new_X_with_name = np.c_[data.iloc[:,0],new_X]

    new_texts = []
    for i in range(len(set(y))):
        # print(i)
        # print(new_X_with_name[idx[i],1])
        plt.scatter(new_X[idx[i],0], new_X[idx[i],1], color=color_map[i], label=i+1)
        for j in range(len(new_X_with_name[idx[i], 0])):
            new_X_with_name = np.array(new_X_with_name)
            # new_X_with_name[idx[i], 0][j] = [eval(a) for a in (new_X_with_name[idx[i],0][j])]
            texts = plt.text(new_X_with_name[idx[i], 1][j], new_X_with_name[idx[i], 2][j],
                             (new_X_with_name[idx[i], 0][j]), fontsize=10, verticalalignment="top",
                             horizontalalignment="right")
            new_texts.append(texts)
        adjust_text(new_texts,
                    only_move={'texts': 'x', },
                    # arrowprops=dict(arrowstyle="-",
                    #                 color='black',
                    #                 lw=0.5))
                    )

    plt.xlabel('new_feature_1')
    plt.ylabel('new_feature_2')
    plt.title('PCA')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    data = deal_data()
    data = data_centering(data)
    data = Normalize(data)
    newData = PCA_method(data)
    draw_data(newData,data)












