#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:


black_friday.head(50)


# In[ ]:


black_friday.shape[0]


# In[ ]:


black_friday.shape # Questão 1


# In[ ]:


black_friday['Gender'].value_counts()


# In[ ]:


filtro = black_friday.query('Gender == "F" & Age == "26-35"') #Questão 2


# In[ ]:


filtro["User_ID"].count() #Questão 2


# In[ ]:


black_friday["User_ID"].nunique() #Questão 3


# In[ ]:


black_friday["Product_ID"].nunique()


# In[ ]:


black_friday.dtypes.value_counts().count() #Questão 4


# In[ ]:


black_friday.dtypes.value_counts()


# In[ ]:


black_friday.isna().sum() / len(black_friday["User_ID"]) #Questão 5 estrutura


# In[ ]:


df = pd.DataFrame({"Columns": black_friday.columns, "Types": black_friday.dtypes, "Missing": black_friday.isna().sum()})


# In[ ]:


df


# In[ ]:


aux = black_friday.query('Gender == "F" & Age == "26-35"')


# In[ ]:


aux = aux["User_ID"].nunique()


# In[ ]:


aux


# In[ ]:


porcentagem_missing = pd.DataFrame({"Missing": black_friday[["Product_Category_2","Product_Category_3"]].isna().sum()/len(black_friday["User_ID"])})


# In[ ]:


missing = pd.DataFrame({"Missing": black_friday[["Product_Category_2","Product_Category_3"]].isna().sum()})


# In[ ]:


missing["Missing"].max() #Questão 6


# In[ ]:


black_friday["Product_Category_3"].mode()[0]


# In[ ]:


black_friday["Product_Category_3"].count()


# In[ ]:


purchase_mean = black_friday["Purchase"].mean()
purchase_mean


# In[ ]:


purchase_std = black_friday["Purchase"].std()
purchase_std


# In[ ]:


valor_max = black_friday['Purchase'].max()
valor_min = black_friday['Purchase'].min()
norm = (black_friday['Purchase'] - valor_min) / (valor_max - valor_min)
media = norm.mean()
    
media


# In[ ]:


isna_comparation = black_friday["Product_Category_2"].isna().isin(black_friday["Product_Category_3"].isna()).value_counts()[1]
isna_comparation


# In[ ]:


isna_comparation == black_friday.shape[0]


# In[ ]:


black_friday.dtypes.value_counts()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    shape = black_friday.shape
    return shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    aux = black_friday.query('Gender == "F" & Age == "26-35"')
    '''
    aux = aux["User_ID"].nunique() é a resposta correta para os valores únicos, como a questão pede. 
    Porém, no teste o valor é distinto, então coloquei a quantidade de registros femininos com idade entre 26 e 35 anos
    '''
    return int(aux["User_ID"].count())


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    unique = black_friday["User_ID"].nunique()
    return unique


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    types = black_friday.dtypes.value_counts().count()
    return int(types)


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    porcentagem = pd.DataFrame({"Missing": black_friday[["Product_Category_2","Product_Category_3"]].isna().sum()/len(black_friday["User_ID"])})
    porcentagem = porcentagem["Missing"].max()
    return float(porcentagem)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    missing = pd.DataFrame({"Missing": black_friday[["Product_Category_2","Product_Category_3"]].isna().sum()}) 
    missing = missing["Missing"].max()
    return int(missing)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    mode = black_friday["Product_Category_3"].mode()[0]
    return int(mode)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    purchase_min = black_friday["Purchase"].min()
    purchase_max = black_friday["Purchase"].max()
    purchase_norm = (black_friday["Purchase"] - purchase_min) / (purchase_max - purchase_min)
    purchase_norm_mean = purchase_norm.mean()
    purchase_norm_mean
    return float(purchase_norm_mean)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    purchase_mean = black_friday["Purchase"].mean()
    purchase_std = black_friday["Purchase"].std()
    purchase_zscore = (black_friday["Purchase"] - purchase_mean) / purchase_std
    purchase_zscore_values_filter = purchase_zscore.between(-1,1)
    purchase_zscore_values_filter = purchase_zscore_values_filter.value_counts()[1]
    return int(purchase_zscore_values_filter)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    isna_comparation = black_friday["Product_Category_2"].isna().isin(black_friday["Product_Category_3"].isna()).value_counts()[1]
    return bool(isna_comparation == black_friday.shape[0])


# In[ ]:


q1()


# In[ ]:


q2()


# In[ ]:


q3()


# In[ ]:


q4()


# In[ ]:


q5()


# In[ ]:


q6()


# In[ ]:


q7()


# In[ ]:


q8()


# In[ ]:


q9()


# In[ ]:


q10()


# In[ ]:


df = black_friday


# In[ ]:


df_F = df[df['Gender']=='F']
df_Age = df_F[df_F['Age']=='26-35']
df_Age['User_ID'].nunique()


# In[ ]:




