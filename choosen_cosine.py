from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
vec1 = np.array([[1,1,0,1,1], [1,1,0,1,1], [1,1,0,1,1],])
vec2 = np.array([[0,1,0,1,1], [0,1,0,1,1]])
#print(cosine_similarity([vec1, vec2]))


import pandas as pd
logging.basicConfig(level=logging.INFO, filename='calc.log')

N = 91560
def load_data():
    d = ["代码", "提升", "整改", "告警", "驱动", "软件", "设备", "v8r11c00", "v8r10c10", "用户", "接口", "流控",
         "子卡", "配置", "清理", "支持", "st", "定位", "特性", "单板", "测试", "加固", "修改",
         "v8r11c10", "开发", "业务", "质量", "项目", "问题", "迭代"]
    df = pd.read_excel('wangwenjia.xlsx', index_col=0, nrows=N)
    df = df[(df['category'] == 'FEI_DOP_588X') | (df['category'] =='BR_UM') |
        (df['category'] =='BOARD_DRIVER') | (df['category'] =='公共模块') | (df['category'] =='BR_AAA') |
        (df['category'] =='VSM') | (df['category'] =='FEI_DOP_COMMON') | (df['category'] =='FEI_DOP_UTIL')]
    df = df[~df.text.isnull()]
    pars = np.zeros((len(d), df.shape[0]))
    
    i = 0
    for index, row in df.iterrows():
        j = 0
        for item in d:
            pars[j, i] = row['text'].count(item)
            j = j + 1
        i = i + 1
            
    u, s, v = np.linalg.svd(pars)
    s = np.diag(s)
    print('pars shape: ', pars.T.shape, 'u shape: ', u.shape, 's shape: ', np.linalg.pinv(s).shape)
    coordinates = pars.T @ u @ np.linalg.pinv(s)
    
    labels = ['FEI_DOP_588X', 'BOARD_DRIVER', 'BR_UM', '公共模块', 'BR_AAA', 'VSM', 'FEI_DOP_COMMON',
              'FEI_DOP_UTIL']
    
    for m in range(6):
        k = m
        if k < 1:
            k = 1
        for n in range(k, 6):
            
            df1 = df[(df['category'] == labels[m])]
            df2 = df[(df['category'] ==labels[n])]
            
            vec1 = np.array([[]])
            i = 0
            for index, row in df1.iterrows():
                arr = []
                for j in range(len(d)):
                    arr.append(coordinates[i][j])
                
                if len(vec1[0]) == 0:
                    vec1 = [arr]
                else:
                    vec1 = np.concatenate((vec1, [arr]))
                    
                i = i + 1
              
            vec2 = np.array([[]])
            i = 0
            for index, row in df2.iterrows():
                arr = []
                for j in range(len(d)):
                    arr.append(coordinates[i][j])
                
                if len(vec2[0]) == 0:
                    vec2 = [arr]
                else:
                    vec2 = np.concatenate((vec2, [arr]))
                    
                i = i + 1
            
            sim = cosine_similarity(vec1, vec2)
            cnt = 0
            
            P = sim.shape[0] if sim.shape[0] < sim.shape[1] else sim.shape[1]
            Q = sim.shape[0] if sim.shape[0] >= sim.shape[1] else sim.shape[1]
            if sim.shape[0] >= sim.shape[1]:
                sim = sim.T
            for i in range(P):
                for j in range(Q):
                    if sim[i, j] > 0.9:
                        cnt = cnt + 1
                        break
                        #print(i,j, cnt)
            logging.info('{}x{}  m={}, n={}, cnt={}'.format(sim.shape[0], sim.shape[1], m, n, 100.0 * cnt / P ))
            
            
            
    #print(cosine_similarity([a[0]], [a[0]]))

def analyze():
    df = pd.read_excel('choosen.xlsx', index_col=0, nrows=91560)
    print(df.head())
    drop_rows = []
    for index, row in df.iterrows():
        s = 0
        for q in range(30):
              s = s + row[q]
        if s < 0.5:
            drop_rows.append(index)
    df = df.drop(drop_rows)
    
    labels = ['FEI_DOP_588X', 'BOARD_DRIVER', 'BR_UM', '公共模块', 'BR_AAA', 'VSM']
    
    #| (df['MODULE'] =='BOARD_DRIVER') |
        #(df['MODULE'] =='BR_UM') | (df['MODULE'] =='公共模块') | (df['MODULE'] =='BR_AAA') | (df['MODULE'] =='VSM')]

    for m in range(6):
        for n in range(m , 6):
            logging.info('m={}, n={}'.format(m, n))
            df1 = df[(df['MODULE'] == labels[m])]
            df2 = df[(df['MODULE'] ==labels[n])]
            i = 1
            for col in df1.columns[:-1]:
                print(i, df1[col].sum())
                i = i + 1
            
            vec1 = np.array([[]])
            for index, row in df1.iterrows():
                arr = []
                for j in range(30):
                    arr.append(row[j])
                
                if len(vec1[0]) == 0:
                    vec1 = [arr]
                else:
                    vec1 = np.concatenate((vec1, [arr]))
              
            vec2 = np.array([[]])
            for index, row in df2.iterrows():
                arr = []
                for j in range(30):
                    arr.append(row[j])
                
                if len(vec2[0]) == 0:
                    vec2 = [arr]
                else:
                    vec2 = np.concatenate((vec2, [arr]))
                    
            logging.info(cosine_similarity(vec1, vec2))
    
    
    
def calculate():
    df = pd.read_csv('D:\\Data\\4.csv', sep='\t', header=0, index_col =0)
    df_full = df.copy(deep=True)
    df = df.drop(df.columns[3338], axis=1)
    df = df.drop(df.columns[3337], axis=1)
    df = df.drop(df.columns[3336], axis=1)
    df = df.drop(df.columns[3335], axis=1)
    df = df.drop(df.columns[3334], axis=1)
    df = df.drop(df.columns[3333], axis=1)
    arr = []
    drop_rows = []
    for index, row in df_full.iterrows():
        s = 0
        for q in range(6):
              s = s + row[3338-q]
        if s < 0.5:
            drop_rows.append(index)
              
    for index, row in df.iterrows():            
        for j in range(len(df.columns)):
            if len(arr) <= j:
                arr.append({'col': j, 'sum': 0})
            arr[j] = {'col': j, 'sum': arr[j]['sum'] + row[j]}
    df_full = df_full.drop(drop_rows)
    df = df.drop(drop_rows)    
    arr1 = sorted(arr, key=lambda k: k['sum'], reverse=True)
    
    df_selected = df.iloc[: , list(map(lambda x: x['col'], arr1[0:10]))].copy(deep=True) 
    
    df4 = df_full.iloc[: , [3338, 3337, 3336, 3335, 3334, 3333]].copy(deep=True) 
    df_selected = df_selected.join(df4)

    
    label1 = 'l1'
    label2 = 'l2'
    df1 = df_selected[ (df_selected.toxic == 1) | (df_selected.toxic == 1)]
    df2 = df_selected[ (df_selected.severe_toxic == 1) | (df_selected.severe_toxic == 1)]

    vec1 = np.array([[]])
    for index, row in df1.iterrows():
        arr = []
        for j in range(10):
            arr.append(row[j])
        
        if len(vec1[0]) == 0:
            vec1 = [arr]
        else:
            vec1 = np.concatenate((vec1, [arr]))
            
            
    vec2 = np.array([[]])
    for index, row in df2.iterrows():
        arr = []
        for j in range(10):
            arr.append(row[j])
        
        if len(vec2[0]) == 0:
            vec2 = [arr]
        else:
            vec2 = np.concatenate((vec2, [arr]))
            
    print(cosine_similarity(vec1, vec2))
    
    
#calculate()
load_data()