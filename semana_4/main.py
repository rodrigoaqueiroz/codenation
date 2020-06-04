#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# Rascunho Questão 1

# In[4]:


dataframe.describe()


# In[7]:


dataframe.head()


# In[66]:


q1_norm = dataframe["normal"].quantile(0.25)
q2_norm = dataframe["normal"].quantile(0.50)
q3_norm = dataframe["normal"].quantile(0.75)
print(q1_norm)
print(q2_norm)
print(q3_norm)


# In[68]:


q1_binom = dataframe["binomial"].quantile(0.25)
q2_binom = dataframe["binomial"].quantile(0.50)
q3_binom = dataframe["binomial"].quantile(0.75)
print(q1_binom)
print(q2_binom)
print(q3_binom)


# In[69]:


quartiles = pd.DataFrame({"Q1": [q1_norm, q1_binom], "Q2": [q2_norm, q2_binom], "Q3": [q3_norm, q3_binom]}, index = ["Normal", "Binomial"])
quartiles


# In[227]:


diff_q1 = round(quartiles["Q1"][0] - quartiles["Q1"][1], 4)
diff_q2 = round(quartiles["Q2"][0] - quartiles["Q2"][1], 4)
diff_q3 = round(quartiles["Q3"][0] - quartiles["Q3"][1], 4)
print(diff_q1, diff_q2, diff_q3)


# Rascunho Questão 2

# In[99]:


ECDF(dataframe["normal"])


# In[100]:


ecdf = ECDF(dataframe["normal"])
mean = dataframe["normal"].mean()
std = dataframe["normal"].std()
interval_normal = [mean - std, mean + std]
prob_interval = round(ecdf(interval_normal)[1] - ecdf(interval_normal)[0], 3)
prob_interval


# In[105]:


ecdf = ECDF(dataframe["normal"])
mean = dataframe["normal"].mean()
std = dataframe["normal"].std()
interval_normal = [mean - 2*std, mean + 2*std]
prob_interval = round(ecdf(interval_normal)[1] - ecdf(interval_normal)[0], 3)
prob_interval


# In[106]:


ecdf = ECDF(dataframe["normal"])
mean = dataframe["normal"].mean()
std = dataframe["normal"].std()
interval_normal = [mean - 3*std, mean + 3*std]
prob_interval = round(ecdf(interval_normal)[1] - ecdf(interval_normal)[0], 3)
prob_interval


# In[108]:


ecdf = ECDF(dataframe["normal"])
mean = dataframe["normal"].mean()
std = dataframe["normal"].std()
interval_normal = [mean - 4*std, mean + 4*std]
prob_interval = round(ecdf(interval_normal)[1] - ecdf(interval_normal)[0], 3)
prob_interval


# O valor é muito próximo do valor téorico, onde 68.26% está entre -1 e 1 de std; 95.44% entre -2 e 2 std e 99.73% entre -3 e 3.

# Rascunho Questão 3

# In[116]:


m_norm = dataframe["normal"].mean()
v_norm = dataframe["normal"].var()
m_binom = dataframe["binomial"].mean()
v_binom = dataframe["binomial"].var()
diff = round(m_binom - m_norm,3), round(v_binom - v_norm,3)
diff


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[230]:


def q1():
    quartis = pd.DataFrame({"Q1": [q1_norm, q1_binom], "Q2": [q2_norm, q2_binom], "Q3": [q3_norm, q3_binom]}, index = ["Normal", "Binomial"])
    diff_q1 = round(quartis["Q1"][0] - quartis["Q1"][1], 3)
    diff_q2 = round(quartis["Q2"][0] - quartis["Q2"][1], 3)
    diff_q3 = round(quartis["Q3"][0] - quartis["Q3"][1], 3)
    diff_per_quartile = (diff_q1, diff_q2, diff_q3)
    return diff_per_quartile


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[224]:


def q2():
    ecdf = ECDF(dataframe["normal"])
    mean = dataframe["normal"].mean()
    std = dataframe["normal"].std()
    interval_norm = [mean - std, mean + std]
    prob_interval = ecdf(interval_norm)[1] - ecdf(interval_norm)[0]
    prob_interval_float_format = float('{:.3f}'.format(prob_interval))
    return prob_interval_float_format


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[117]:


def q3():
    m_norm = dataframe["normal"].mean()
    v_norm = dataframe["normal"].var()
    m_binom = dataframe["binomial"].mean()
    v_binom = dataframe["binomial"].var()
    diff = round(m_binom - m_norm,3), round(v_binom - v_norm,3)
    return diff


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[135]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[136]:


# Sua análise da parte 2 começa aqui.
stars.head()


# In[134]:


stars["target"].value_counts()


# In[127]:


stars.isnull().count()


# In[130]:


stars.describe()


# In[243]:


stars.shape


# In[276]:


aux = stars.query("target == 0")["mean_profile"]
false_pulsar_mean_profile_standardized = ((aux - aux.mean()))/aux.std()
false_pulsar_mean_profile_standardized


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[289]:


def q4():
    aux = stars.query("target == 0")["mean_profile"]
    false_pulsar_mean_profile_standardized = (aux - aux.mean())/aux.std()
    norm = sct.norm.ppf([0.80, 0.90, 0.95])
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    prob = ecdf(norm)
    return tuple(prob.round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[291]:


def q5():
    aux = stars.query("target == 0")["mean_profile"]
    false_pulsar_mean_profile_standardized = (aux - aux.mean())/aux.std()
    norm = sct.norm.ppf([0.25, 0.50, 0.75])
    false_pulsar_mean_profile_standardized_q1 = false_pulsar_mean_profile_standardized.quantile(0.25)
    false_pulsar_mean_profile_standardized_q2 = false_pulsar_mean_profile_standardized.quantile(0.50)
    false_pulsar_mean_profile_standardized_q3 = false_pulsar_mean_profile_standardized.quantile(0.75)
    false_pulsar_mean_profile_standardized_quartiles = [false_pulsar_mean_profile_standardized_q1,
                                                    false_pulsar_mean_profile_standardized_q2, 
                                                    false_pulsar_mean_profile_standardized_q3]
    false_pulsar_mean_profile_standardized_quartiles
    diff = false_pulsar_mean_profile_standardized_quartiles - norm
    return tuple(diff.round(3)) 


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# In[ ]:




