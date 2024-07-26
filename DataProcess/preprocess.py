import os
import pandas as pd
import copy
#Convert to datasets arranged in a specific order, and merge all datasets in the folder.

def getData(dataframe,list):

    col=dataframe.columns.values     
    new_list = copy.deepcopy(list)   
    for i,data in enumerate(col):     
        for j,item in enumerate(new_list): 
            if data==item:           
                del new_list[j]
                break
            elif j==len(new_list)-1 and data!=item:  
                dataframe.drop(columns=data,inplace=True)

    zero_df = pd.DataFrame(0,index=range(len(dataframe)),columns=new_list)  
    res_heng= pd.concat([dataframe,zero_df], axis=1)                        
    data = res_heng.loc[:, list].values 
    data=data.astype(str)
    return data

folder_path = 'D:\\load\\GEO_pretrain'
combined_data = pd.DataFrame()
gene_list = []

with open('D:\\load\\Gene2vec-master\\pre_trained_emb\\gene2vec_dim_200_iter_9.txt', 'r') as file:

    for line in file:

        columns = line.split()
        gene_name = columns[0]  
        gene_list.append(gene_name)  



for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        

        data = pd.read_csv(file_path)
        

        processed_data = getData(data,gene_list)
        processed_df = pd.DataFrame(processed_data)

        combined_data = pd.concat([combined_data, processed_df], ignore_index=True)
           
combined_data.to_csv('D:\\load\\GEO_pretrain\\result.csv', index=False)



#Preprocess and binning 
import numpy as np
import scanpy as sc

df = pd.read_csv('D:\\load\\GEO_pretrain\\result.csv')

df = df.astype(float)
adata = sc.AnnData(df)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata, base=2)
processed_data = adata.X

min_value = processed_data[processed_data > 0].min().min()
max_value = processed_data.max().max()

bin_width = (max_value - min_value) / 20000

bin_edges = np.linspace(min_value, max_value, num=20000)

bins = np.digitize(processed_data, bin_edges)

bins=bins.astype(str)    
                                   
output_path = 'D:\\load\\GEO_pretrain\\result_bin.csv'
np.savetxt(output_path, bins, delimiter=',', fmt='%s')

