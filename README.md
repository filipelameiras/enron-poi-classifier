
# 1. Introdução

Este projeto tem como objetivo criar um algoritmo que identifique os funcionários da Enron que podem ter cometido fraude baseando-se no conjunto de dados público entitulado "Enron financial and email".

Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa.

Este documento visa explicar minha linha de raciocínio e responder as questões, indagadas pela Udacity, para avaliação deste projeto. As perguntas encontram-se neste [link.](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true)

# 2. Conjunto de dados

Os dados foram disponibilizados em um dicionário, onde cada par chave-valor corresponde a uma pessoa. A chave do dicionário é o nome da pessoa, e o valor é outro dicionário, que contém o nome de todos os atributos e seus valores para aquela pessoa. Os atributos nos dados possuem basicamente três tipos: atributos financeiros, de email e rótulos POI (pessoa de interesse).

>**atributos financeiros:** ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))

>**atributos de email:** ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)

>**rótulo POI:** [‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro)

# 3. Seleção de atributos (features)

Antes de selecionar os atributos iniciais a serem usados, deve-se primeiro realizar uma análise sobre a situação dos dados disponibilizados. O objetivo dessa análise é descobrir como os dados estão estruturados, se existem dados faltantes, problemas com os tipos de dados ou algum outro tipo de problema que inviabilize o uso de alguma feature inicialmente.

>Importando as bibliotecas necessárias para a análise


```python
import sys
import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
```

>Carregando os dados...


```python
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
```

>Tamanho dos dados...


```python
print("{0} linhas, {1} colunas".format(df.shape[0],df.shape[1]))
```

    146 linhas, 21 colunas


>Visualizando os dados...


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALLEN PHILLIP K</th>
      <td>201955</td>
      <td>2902</td>
      <td>2869717</td>
      <td>4484442</td>
      <td>NaN</td>
      <td>4175000</td>
      <td>phillip.allen@enron.com</td>
      <td>-126027</td>
      <td>-3081055</td>
      <td>1729541</td>
      <td>...</td>
      <td>47</td>
      <td>1729541</td>
      <td>2195</td>
      <td>152</td>
      <td>65</td>
      <td>False</td>
      <td>304805</td>
      <td>1407</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BADUM JAMES P</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>178980</td>
      <td>182466</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257817</td>
      <td>...</td>
      <td>NaN</td>
      <td>257817</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BANNANTINE JAMES M</th>
      <td>477</td>
      <td>566</td>
      <td>NaN</td>
      <td>916197</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>james.bannantine@enron.com</td>
      <td>-560222</td>
      <td>-5104</td>
      <td>5243487</td>
      <td>...</td>
      <td>39</td>
      <td>4046157</td>
      <td>29</td>
      <td>864523</td>
      <td>0</td>
      <td>False</td>
      <td>NaN</td>
      <td>465</td>
      <td>1757552</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102</td>
      <td>NaN</td>
      <td>1295738</td>
      <td>5634343</td>
      <td>NaN</td>
      <td>1200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055</td>
      <td>10623258</td>
      <td>...</td>
      <td>NaN</td>
      <td>6680544</td>
      <td>NaN</td>
      <td>2660303</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055</td>
      <td>NaN</td>
      <td>3942714</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAY FRANKLIN R</th>
      <td>239671</td>
      <td>NaN</td>
      <td>260455</td>
      <td>827696</td>
      <td>NaN</td>
      <td>400000</td>
      <td>frank.bay@enron.com</td>
      <td>-82782</td>
      <td>-201641</td>
      <td>63014</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>145796</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAZELIDES PHILIP J</th>
      <td>80818</td>
      <td>NaN</td>
      <td>684694</td>
      <td>860136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1599641</td>
      <td>...</td>
      <td>NaN</td>
      <td>1599641</td>
      <td>NaN</td>
      <td>874</td>
      <td>NaN</td>
      <td>False</td>
      <td>93750</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BECK SALLY W</th>
      <td>231330</td>
      <td>7315</td>
      <td>NaN</td>
      <td>969068</td>
      <td>NaN</td>
      <td>700000</td>
      <td>sally.beck@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>126027</td>
      <td>...</td>
      <td>144</td>
      <td>NaN</td>
      <td>4343</td>
      <td>566</td>
      <td>386</td>
      <td>False</td>
      <td>NaN</td>
      <td>2639</td>
      <td>126027</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BELDEN TIMOTHY N</th>
      <td>213999</td>
      <td>7991</td>
      <td>2144013</td>
      <td>5501630</td>
      <td>NaN</td>
      <td>5249999</td>
      <td>tim.belden@enron.com</td>
      <td>NaN</td>
      <td>-2334434</td>
      <td>1110705</td>
      <td>...</td>
      <td>228</td>
      <td>953136</td>
      <td>484</td>
      <td>210698</td>
      <td>108</td>
      <td>True</td>
      <td>NaN</td>
      <td>5521</td>
      <td>157569</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BELFER ROBERT</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-102500</td>
      <td>102500</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44093</td>
      <td>NaN</td>
      <td>-44093</td>
      <td>...</td>
      <td>NaN</td>
      <td>3285</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3285</td>
    </tr>
    <tr>
      <th>BERBERIAN DAVID</th>
      <td>216582</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>228474</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>david.berberian@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2493616</td>
      <td>...</td>
      <td>NaN</td>
      <td>1624396</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>869220</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>



>Vizualizando tipos de dados e dados faltantes...


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 21 columns):
    salary                       146 non-null object
    to_messages                  146 non-null object
    deferral_payments            146 non-null object
    total_payments               146 non-null object
    loan_advances                146 non-null object
    bonus                        146 non-null object
    email_address                146 non-null object
    restricted_stock_deferred    146 non-null object
    deferred_income              146 non-null object
    total_stock_value            146 non-null object
    expenses                     146 non-null object
    from_poi_to_this_person      146 non-null object
    exercised_stock_options      146 non-null object
    from_messages                146 non-null object
    other                        146 non-null object
    from_this_person_to_poi      146 non-null object
    poi                          146 non-null bool
    long_term_incentive          146 non-null object
    shared_receipt_with_poi      146 non-null object
    restricted_stock             146 non-null object
    director_fees                146 non-null object
    dtypes: bool(1), object(20)
    memory usage: 24.1+ KB


Primeiro problema encontrado: Muitos atribuitos faltantes estão com os valores 'NAN' no formato texto. Corrigir este problema pois pode influenciar na estatística. 


```python
df = df.replace('NaN', np.NaN)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 146 entries, ALLEN PHILLIP K to YEAP SOON
    Data columns (total 21 columns):
    salary                       95 non-null float64
    to_messages                  86 non-null float64
    deferral_payments            39 non-null float64
    total_payments               125 non-null float64
    loan_advances                4 non-null float64
    bonus                        82 non-null float64
    email_address                111 non-null object
    restricted_stock_deferred    18 non-null float64
    deferred_income              49 non-null float64
    total_stock_value            126 non-null float64
    expenses                     95 non-null float64
    from_poi_to_this_person      86 non-null float64
    exercised_stock_options      102 non-null float64
    from_messages                86 non-null float64
    other                        93 non-null float64
    from_this_person_to_poi      86 non-null float64
    poi                          146 non-null bool
    long_term_incentive          66 non-null float64
    shared_receipt_with_poi      86 non-null float64
    restricted_stock             110 non-null float64
    director_fees                17 non-null float64
    dtypes: bool(1), float64(19), object(1)
    memory usage: 24.1+ KB


Agora vemos que somente o atributo 'poi' está completo.

>Quantas variáveis faltante cada funcionário possui?


```python
df.isnull().sum(axis=1).sort_values(ascending=False)
```




    LOCKHART EUGENE E                20
    GRAMM WENDY L                    18
    WROBEL BRUCE                     18
    WHALEY DAVID A                   18
    THE TRAVEL AGENCY IN THE PARK    18
    WAKEHAM JOHN                     17
    WODRASKA JOHN                    17
    CLINE KENNETH W                  17
    GILLIS JOHN                      17
    SCRIMSHAW MATTHEW                17
    SAVAGE FRANK                     17
    MENDELSOHN JOHN                  16
    YEAP SOON                        16
    CHRISTODOULOU DIOMEDES           16
    PEREIRA PAULO V. FERRAZ          16
    BLAKE JR. NORMAN P               16
    LOWRY CHARLES P                  16
    CHAN RONNIE                      16
    MEYER JEROME J                   16
    GATHMANN WILLIAM D               16
    WINOKUR JR. HERBERT S            16
    FUGH JOHN L                      16
    URQUHART JOHN A                  16
    NOLES JAMES L                    15
    WALTERS GARETH W                 15
    BADUM JAMES P                    15
    LEMAISTRE CHARLES                15
    DUNCAN JOHN H                    15
    GRAY RODNEY                      15
    PRENTICE JAMES                   14
                                     ..
    THORN TERENCE H                   5
    SKILLING JEFFREY K                5
    SHELBY REX                        5
    SHANKMAN JEFFREY A                5
    MURRAY JULIA H                    5
    DURAN WILLIAM D                   5
    MCCONNELL MICHAEL S               5
    MCMAHON JEFFREY                   5
    FALLON JAMES B                    5
    FITZGERALD JAY L                  5
    KOENIG MARK E                     5
    KEAN STEVEN J                     5
    GARLAND C KEVIN                   5
    GLISAN JR BEN F                   5
    LAVORATO JOHN J                   5
    RIEKER PAULA H                    4
    BELDEN TIMOTHY N                  4
    SHARP VICTORIA T                  4
    RICE KENNETH D                    4
    WASAFF GEORGE                     4
    OLSON CINDY K                     4
    HANNON KEVIN P                    4
    MULLER MARK S                     4
    BUY RICHARD B                     4
    PIPER GREGORY F                   3
    DERRICK JR. JAMES V               3
    LAY KENNETH L                     2
    HAEDICKE MARK E                   2
    FREVERT MARK A                    2
    ALLEN PHILLIP K                   2
    Length: 146, dtype: int64



Vemos que a pessoa **LOCKHART EUGENE E** não possui nenhum atributo além do 'poi'. Então ele deve ser retirado da base já que não possui nenhuma informação.


```python
df.drop('LOCKHART EUGENE E', inplace=True)
```

>**THE TRAVEL AGENCY IN THE PARK** não parece ser uma pessoa. Quais os atributos que possui?


```python
df.loc['THE TRAVEL AGENCY IN THE PARK']
```




    salary                          NaN
    to_messages                     NaN
    deferral_payments               NaN
    total_payments               362096
    loan_advances                   NaN
    bonus                           NaN
    email_address                   NaN
    restricted_stock_deferred       NaN
    deferred_income                 NaN
    total_stock_value               NaN
    expenses                        NaN
    from_poi_to_this_person         NaN
    exercised_stock_options         NaN
    from_messages                   NaN
    other                        362096
    from_this_person_to_poi         NaN
    poi                           False
    long_term_incentive             NaN
    shared_receipt_with_poi         NaN
    restricted_stock                NaN
    director_fees                   NaN
    Name: THE TRAVEL AGENCY IN THE PARK, dtype: object



Devemos excluir **THE TRAVEL AGENCY IN THE PARK**. Pois não se trata de uma pessoa.


```python
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)
```

>Quantos e quais funcionários são classificados como POI?


```python
df.poi.value_counts()
```




    False    126
    True      18
    Name: poi, dtype: int64




```python
df.query("poi==True")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BELDEN TIMOTHY N</th>
      <td>213999.0</td>
      <td>7991.0</td>
      <td>2144013.0</td>
      <td>5501630.0</td>
      <td>NaN</td>
      <td>5249999.0</td>
      <td>tim.belden@enron.com</td>
      <td>NaN</td>
      <td>-2334434.0</td>
      <td>1110705.0</td>
      <td>...</td>
      <td>228.0</td>
      <td>953136.0</td>
      <td>484.0</td>
      <td>210698.0</td>
      <td>108.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>5521.0</td>
      <td>157569.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BOWEN JR RAYMOND M</th>
      <td>278601.0</td>
      <td>1858.0</td>
      <td>NaN</td>
      <td>2669589.0</td>
      <td>NaN</td>
      <td>1350000.0</td>
      <td>raymond.bowen@enron.com</td>
      <td>NaN</td>
      <td>-833.0</td>
      <td>252055.0</td>
      <td>...</td>
      <td>140.0</td>
      <td>NaN</td>
      <td>27.0</td>
      <td>1621.0</td>
      <td>15.0</td>
      <td>True</td>
      <td>974293.0</td>
      <td>1593.0</td>
      <td>252055.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CALGER CHRISTOPHER F</th>
      <td>240189.0</td>
      <td>2598.0</td>
      <td>NaN</td>
      <td>1639297.0</td>
      <td>NaN</td>
      <td>1250000.0</td>
      <td>christopher.calger@enron.com</td>
      <td>NaN</td>
      <td>-262500.0</td>
      <td>126027.0</td>
      <td>...</td>
      <td>199.0</td>
      <td>NaN</td>
      <td>144.0</td>
      <td>486.0</td>
      <td>25.0</td>
      <td>True</td>
      <td>375304.0</td>
      <td>2188.0</td>
      <td>126027.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CAUSEY RICHARD A</th>
      <td>415189.0</td>
      <td>1892.0</td>
      <td>NaN</td>
      <td>1868758.0</td>
      <td>NaN</td>
      <td>1000000.0</td>
      <td>richard.causey@enron.com</td>
      <td>NaN</td>
      <td>-235000.0</td>
      <td>2502063.0</td>
      <td>...</td>
      <td>58.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>307895.0</td>
      <td>12.0</td>
      <td>True</td>
      <td>350000.0</td>
      <td>1585.0</td>
      <td>2502063.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>COLWELL WESLEY</th>
      <td>288542.0</td>
      <td>1758.0</td>
      <td>27610.0</td>
      <td>1490344.0</td>
      <td>NaN</td>
      <td>1200000.0</td>
      <td>wes.colwell@enron.com</td>
      <td>NaN</td>
      <td>-144062.0</td>
      <td>698242.0</td>
      <td>...</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>101740.0</td>
      <td>11.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>1132.0</td>
      <td>698242.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>DELAINEY DAVID W</th>
      <td>365163.0</td>
      <td>3093.0</td>
      <td>NaN</td>
      <td>4747979.0</td>
      <td>NaN</td>
      <td>3000000.0</td>
      <td>david.delainey@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3614261.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>2291113.0</td>
      <td>3069.0</td>
      <td>1661.0</td>
      <td>609.0</td>
      <td>True</td>
      <td>1294981.0</td>
      <td>2097.0</td>
      <td>1323148.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>FASTOW ANDREW S</th>
      <td>440698.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2424083.0</td>
      <td>NaN</td>
      <td>1300000.0</td>
      <td>andrew.fastow@enron.com</td>
      <td>NaN</td>
      <td>-1386055.0</td>
      <td>1794412.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>277464.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>1736055.0</td>
      <td>NaN</td>
      <td>1794412.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>GLISAN JR BEN F</th>
      <td>274975.0</td>
      <td>873.0</td>
      <td>NaN</td>
      <td>1272284.0</td>
      <td>NaN</td>
      <td>600000.0</td>
      <td>ben.glisan@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>778546.0</td>
      <td>...</td>
      <td>52.0</td>
      <td>384728.0</td>
      <td>16.0</td>
      <td>200308.0</td>
      <td>6.0</td>
      <td>True</td>
      <td>71023.0</td>
      <td>874.0</td>
      <td>393818.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HANNON KEVIN P</th>
      <td>243293.0</td>
      <td>1045.0</td>
      <td>NaN</td>
      <td>288682.0</td>
      <td>NaN</td>
      <td>1500000.0</td>
      <td>kevin.hannon@enron.com</td>
      <td>NaN</td>
      <td>-3117011.0</td>
      <td>6391065.0</td>
      <td>...</td>
      <td>32.0</td>
      <td>5538001.0</td>
      <td>32.0</td>
      <td>11350.0</td>
      <td>21.0</td>
      <td>True</td>
      <td>1617011.0</td>
      <td>1035.0</td>
      <td>853064.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HIRKO JOSEPH</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10259.0</td>
      <td>91093.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>joe.hirko@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30766064.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>30766064.0</td>
      <td>NaN</td>
      <td>2856.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>KOENIG MARK E</th>
      <td>309946.0</td>
      <td>2374.0</td>
      <td>NaN</td>
      <td>1587421.0</td>
      <td>NaN</td>
      <td>700000.0</td>
      <td>mark.koenig@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1920055.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>671737.0</td>
      <td>61.0</td>
      <td>150458.0</td>
      <td>15.0</td>
      <td>True</td>
      <td>300000.0</td>
      <td>2271.0</td>
      <td>1248318.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>KOPPER MICHAEL J</th>
      <td>224305.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2652612.0</td>
      <td>NaN</td>
      <td>800000.0</td>
      <td>michael.kopper@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>985032.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>907502.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>602671.0</td>
      <td>NaN</td>
      <td>985032.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>LAY KENNETH L</th>
      <td>1072321.0</td>
      <td>4273.0</td>
      <td>202911.0</td>
      <td>103559793.0</td>
      <td>81525000.0</td>
      <td>7000000.0</td>
      <td>kenneth.lay@enron.com</td>
      <td>NaN</td>
      <td>-300000.0</td>
      <td>49110078.0</td>
      <td>...</td>
      <td>123.0</td>
      <td>34348384.0</td>
      <td>36.0</td>
      <td>10359729.0</td>
      <td>16.0</td>
      <td>True</td>
      <td>3600000.0</td>
      <td>2411.0</td>
      <td>14761694.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RICE KENNETH D</th>
      <td>420636.0</td>
      <td>905.0</td>
      <td>NaN</td>
      <td>505050.0</td>
      <td>NaN</td>
      <td>1750000.0</td>
      <td>ken.rice@enron.com</td>
      <td>NaN</td>
      <td>-3504386.0</td>
      <td>22542539.0</td>
      <td>...</td>
      <td>42.0</td>
      <td>19794175.0</td>
      <td>18.0</td>
      <td>174839.0</td>
      <td>4.0</td>
      <td>True</td>
      <td>1617011.0</td>
      <td>864.0</td>
      <td>2748364.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RIEKER PAULA H</th>
      <td>249201.0</td>
      <td>1328.0</td>
      <td>214678.0</td>
      <td>1099100.0</td>
      <td>NaN</td>
      <td>700000.0</td>
      <td>paula.rieker@enron.com</td>
      <td>NaN</td>
      <td>-100000.0</td>
      <td>1918887.0</td>
      <td>...</td>
      <td>35.0</td>
      <td>1635238.0</td>
      <td>82.0</td>
      <td>1950.0</td>
      <td>48.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>1258.0</td>
      <td>283649.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SHELBY REX</th>
      <td>211844.0</td>
      <td>225.0</td>
      <td>NaN</td>
      <td>2003885.0</td>
      <td>NaN</td>
      <td>200000.0</td>
      <td>rex.shelby@enron.com</td>
      <td>NaN</td>
      <td>-4167.0</td>
      <td>2493616.0</td>
      <td>...</td>
      <td>13.0</td>
      <td>1624396.0</td>
      <td>39.0</td>
      <td>1573324.0</td>
      <td>14.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>91.0</td>
      <td>869220.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SKILLING JEFFREY K</th>
      <td>1111258.0</td>
      <td>3627.0</td>
      <td>NaN</td>
      <td>8682716.0</td>
      <td>NaN</td>
      <td>5600000.0</td>
      <td>jeff.skilling@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26093672.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>19250000.0</td>
      <td>108.0</td>
      <td>22122.0</td>
      <td>30.0</td>
      <td>True</td>
      <td>1920000.0</td>
      <td>2042.0</td>
      <td>6843672.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>YEAGER F SCOTT</th>
      <td>158403.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>360300.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>scott.yeager@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11884758.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>8308552.0</td>
      <td>NaN</td>
      <td>147950.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3576206.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 21 columns</p>
</div>



>Dos funcionários, quem mais recebeu dinheiro?


```python
df.sort_values(by='total_payments', ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>salary</th>
      <th>to_messages</th>
      <th>deferral_payments</th>
      <th>total_payments</th>
      <th>loan_advances</th>
      <th>bonus</th>
      <th>email_address</th>
      <th>restricted_stock_deferred</th>
      <th>deferred_income</th>
      <th>total_stock_value</th>
      <th>...</th>
      <th>from_poi_to_this_person</th>
      <th>exercised_stock_options</th>
      <th>from_messages</th>
      <th>other</th>
      <th>from_this_person_to_poi</th>
      <th>poi</th>
      <th>long_term_incentive</th>
      <th>shared_receipt_with_poi</th>
      <th>restricted_stock</th>
      <th>director_fees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TOTAL</th>
      <td>26704229.0</td>
      <td>NaN</td>
      <td>32083396.0</td>
      <td>309886585.0</td>
      <td>83925000.0</td>
      <td>97343619.0</td>
      <td>NaN</td>
      <td>-7576788.0</td>
      <td>-27992891.0</td>
      <td>434509511.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>311764000.0</td>
      <td>NaN</td>
      <td>42667589.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>48521928.0</td>
      <td>NaN</td>
      <td>130322299.0</td>
      <td>1398517.0</td>
    </tr>
    <tr>
      <th>LAY KENNETH L</th>
      <td>1072321.0</td>
      <td>4273.0</td>
      <td>202911.0</td>
      <td>103559793.0</td>
      <td>81525000.0</td>
      <td>7000000.0</td>
      <td>kenneth.lay@enron.com</td>
      <td>NaN</td>
      <td>-300000.0</td>
      <td>49110078.0</td>
      <td>...</td>
      <td>123.0</td>
      <td>34348384.0</td>
      <td>36.0</td>
      <td>10359729.0</td>
      <td>16.0</td>
      <td>True</td>
      <td>3600000.0</td>
      <td>2411.0</td>
      <td>14761694.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>FREVERT MARK A</th>
      <td>1060932.0</td>
      <td>3275.0</td>
      <td>6426990.0</td>
      <td>17252530.0</td>
      <td>2000000.0</td>
      <td>2000000.0</td>
      <td>mark.frevert@enron.com</td>
      <td>NaN</td>
      <td>-3367011.0</td>
      <td>14622185.0</td>
      <td>...</td>
      <td>242.0</td>
      <td>10433518.0</td>
      <td>21.0</td>
      <td>7427621.0</td>
      <td>6.0</td>
      <td>False</td>
      <td>1617011.0</td>
      <td>2979.0</td>
      <td>4188667.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BHATNAGAR SANJAY</th>
      <td>NaN</td>
      <td>523.0</td>
      <td>NaN</td>
      <td>15456290.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sanjay.bhatnagar@enron.com</td>
      <td>15456290.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>2604490.0</td>
      <td>29.0</td>
      <td>137864.0</td>
      <td>1.0</td>
      <td>False</td>
      <td>NaN</td>
      <td>463.0</td>
      <td>-2604490.0</td>
      <td>137864.0</td>
    </tr>
    <tr>
      <th>LAVORATO JOHN J</th>
      <td>339288.0</td>
      <td>7259.0</td>
      <td>NaN</td>
      <td>10425757.0</td>
      <td>NaN</td>
      <td>8000000.0</td>
      <td>john.lavorato@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5167144.0</td>
      <td>...</td>
      <td>528.0</td>
      <td>4158995.0</td>
      <td>2585.0</td>
      <td>1552.0</td>
      <td>411.0</td>
      <td>False</td>
      <td>2035380.0</td>
      <td>3962.0</td>
      <td>1008149.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SKILLING JEFFREY K</th>
      <td>1111258.0</td>
      <td>3627.0</td>
      <td>NaN</td>
      <td>8682716.0</td>
      <td>NaN</td>
      <td>5600000.0</td>
      <td>jeff.skilling@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26093672.0</td>
      <td>...</td>
      <td>88.0</td>
      <td>19250000.0</td>
      <td>108.0</td>
      <td>22122.0</td>
      <td>30.0</td>
      <td>True</td>
      <td>1920000.0</td>
      <td>2042.0</td>
      <td>6843672.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MARTIN AMANDA K</th>
      <td>349487.0</td>
      <td>1522.0</td>
      <td>85430.0</td>
      <td>8407016.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>a..martin@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2070306.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>2070306.0</td>
      <td>230.0</td>
      <td>2818454.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>5145434.0</td>
      <td>477.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BAXTER JOHN C</th>
      <td>267102.0</td>
      <td>NaN</td>
      <td>1295738.0</td>
      <td>5634343.0</td>
      <td>NaN</td>
      <td>1200000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1386055.0</td>
      <td>10623258.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>6680544.0</td>
      <td>NaN</td>
      <td>2660303.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>1586055.0</td>
      <td>NaN</td>
      <td>3942714.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>BELDEN TIMOTHY N</th>
      <td>213999.0</td>
      <td>7991.0</td>
      <td>2144013.0</td>
      <td>5501630.0</td>
      <td>NaN</td>
      <td>5249999.0</td>
      <td>tim.belden@enron.com</td>
      <td>NaN</td>
      <td>-2334434.0</td>
      <td>1110705.0</td>
      <td>...</td>
      <td>228.0</td>
      <td>953136.0</td>
      <td>484.0</td>
      <td>210698.0</td>
      <td>108.0</td>
      <td>True</td>
      <td>NaN</td>
      <td>5521.0</td>
      <td>157569.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>DELAINEY DAVID W</th>
      <td>365163.0</td>
      <td>3093.0</td>
      <td>NaN</td>
      <td>4747979.0</td>
      <td>NaN</td>
      <td>3000000.0</td>
      <td>david.delainey@enron.com</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3614261.0</td>
      <td>...</td>
      <td>66.0</td>
      <td>2291113.0</td>
      <td>3069.0</td>
      <td>1661.0</td>
      <td>609.0</td>
      <td>True</td>
      <td>1294981.0</td>
      <td>2097.0</td>
      <td>1323148.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>



Descobrimos mais um problema. **TOTAL** não é um funcionário, e sim um registro que é um somatório de todos os pagamentos feitos a funcionários. Deve ser excluído.


```python
df.drop('TOTAL', inplace=True)
```

>Como este é um caso de fraudes, então uma forma de selecionar os atributos é selecionando aqueles que possium mais outliers. Quais as variáveis que mais possuem outliers?

Obs.: Método utilizado foi o IQR(interquartile range), que pode ser encontrado no [link](http://colingorrie.github.io/outlier-detection.html)


```python
columns = df.columns.values
columns=np.delete(columns,6) #Removi o atributo email_address pois estava dando problema no cálculo

Q1 = df[columns].quantile(0.25)
Q3 = df[columns].quantile(0.75)
IQR = Q3 - Q1
n_outliers = ((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).sum()
n_outliers.sort_values(ascending=False)
```




    total_stock_value            21
    poi                          18
    from_messages                17
    restricted_stock             14
    from_this_person_to_poi      13
    other                        11
    exercised_stock_options      11
    from_poi_to_this_person      11
    total_payments               10
    bonus                        10
    salary                        9
    to_messages                   7
    long_term_incentive           7
    deferral_payments             6
    deferred_income               5
    director_fees                 4
    expenses                      3
    restricted_stock_deferred     2
    shared_receipt_with_poi       2
    loan_advances                 0
    dtype: int64



 Baseado no resultado, Decidi usar os 10 atributos que mais possuem outliers, com exceção do atributo **poi**


```python
feature_list = ['poi', 'total_stock_value', 'from_messages', 'restricted_stock', 'from_this_person_to_poi', 'other', 'exercised_stock_options', 'from_poi_to_this_person', 'total_payments', 'bonus', 'salary']

```

Obs.: Nas próximas sessões iremos usar o algoritmo de seleção de features e comparar com nossa escolha inicial.

# 4. Remoção de outliers

>Na sessão anterior, acabamos removendos os outliers necessários **('TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E')**. Fora isso, como o caso é de fraude, o resto dos outliers não devem ser removidos

# 5. Testando os Classificadores

Escolhi 3 classificadores para realizar esse teste. **Naive Bayes**, **Random Forest**, **Decicion Tree** e **K-means**. Inicialmente vamos testar o desempenho de cada um deles com as features escolhidas no final do item 3 desta análise. Após isso, vamos reavaliar as escolhas das features e fazer um tunning para ver o que consiguimos melhorar no desempenho destes algoritmos. 

## 5.1. Pré-processamentos dos dados


```python
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, main
from sklearn.model_selection import train_test_split
```

>Formatando os dados no formato que os classificadores utilizam 


```python
df.fillna('NaN', inplace=True)
my_dataset = df.to_dict('index')
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```

>Separando o conjunto de dados para treinamento e teste


```python
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
```

## 5.2. Classificadores

Para testar os classificadores vamos usar s funções dump_classifier_and_data, main do arquivo tester, que foi disponibilizado pela Udacity

### 5.2.1. Naive Bayes

>Importando a biblioteca. Criando o classificador e testando.


```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, feature_list)
main()
```

    GaussianNB(priors=None)
    	Accuracy: 0.84807	Precision: 0.38668	Recall: 0.23800	F1: 0.29465	F2: 0.25783
    	Total predictions: 15000	True positives:  476	False positives:  755	False negatives: 1524	True negatives: 12245
    


### 5.2.2. Random Forest

>Importando a biblioteca. Criando o classificador e testando.


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
dump_classifier_and_data(clf, my_dataset, feature_list)
main()
```

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    	Accuracy: 0.85493	Precision: 0.37209	Recall: 0.12800	F1: 0.19048	F2: 0.14733
    	Total predictions: 15000	True positives:  256	False positives:  432	False negatives: 1744	True negatives: 12568
    


### 5.2.3. Decision Tree

>Importando a biblioteca. Criando o classificador e testando.


```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
dump_classifier_and_data(clf, my_dataset, feature_list)
main()
```

    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')
    	Accuracy: 0.79200	Precision: 0.22921	Recall: 0.23700	F1: 0.23304	F2: 0.23540
    	Total predictions: 15000	True positives:  474	False positives: 1594	False negatives: 1526	True negatives: 11406
    


### 5.2.4. K-means

>Importando a biblioteca. Criando o classificador e testando.


```python
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=2, random_state=0).fit(features_train)
dump_classifier_and_data(clf, my_dataset, feature_list)
main()
```

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=0, tol=0.0001, verbose=0)
    	Accuracy: 0.85533	Precision: 0.31360	Recall: 0.07150	F1: 0.11645	F2: 0.08456
    	Total predictions: 15000	True positives:  143	False positives:  313	False negatives: 1857	True negatives: 12687
    


### 5.2.5. Resultados 

>**Naive Bayes:** Accuracy: 0.84807	Precision: 0.38668	Recall: 0.23800	F1: 0.29465	F2: 0.25783

>**Random Forest:** Accuracy: 0.85467	Precision: 0.36527	Recall: 0.12200	F1: 0.18291	F2: 0.14075

>**Decision Tree:** Accuracy: 0.79400	Precision: 0.23492	Recall: 0.24150	F1: 0.23817	F2: 0.24016

>**K-means: Accuracy:** Accuracy: 0.85533	Precision: 0.31360	Recall: 0.07150	F1: 0.11645	F2: 0.084568

Dentre os classificadores testados, o melhor foi o **Naive Bayes**

## 5.3. Reseleção dos atributos e tunning dos classificadores  

Agora vamos realizar os passos abaixo para cada classificador para ver o que conseguimos melhorar

>a) Normalização dos dados, utilizando StandardScaler

>b) Redução de dimensionalidade dos dados, utilizando PCA;

>c) Seleção das Features mais importantes, utilizando SelectKBest;

>d) Otimização, utilizando GridSearchCV; 

>e) Validação cruzada, utilizando StratifiedShuffleSplit.


```python
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
```

>Carregando lista com todas os atributos


```python
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'bonus',
                 'salary',
                 'deferred_income',
                 'long_term_incentive',
                 'restricted_stock',
                 'total_payments',
                 'shared_receipt_with_poi',
                 'loan_advances',
                 'expenses',
                 'from_poi_to_this_person',
                 'other',
                 'from_this_person_to_poi',
                 'director_fees',
                 'to_messages',
                 'deferral_payments',
                 'from_messages',
                 'restricted_stock_deferred'
                ]
```


```python
my_dataset = df.to_dict('index')
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
```


```python
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
```

### 5.3.1. Naive Bayes 

>Pipeline de execução para o classificador


```python
#No caso de Naive Bayes, não foi utilizado o GridSearchCV, para otimização dos kernels params. 
pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('reducer', PCA(random_state=42)),
        ('selector', SelectKBest()),
        ('classifier', GaussianNB())
    ])
```


```python
dump_classifier_and_data(pipe, my_dataset, features_list)
main()
```

    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reducer', PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('selector', SelectKBest(k=10, score_func=<function f_classif at 0x1a1f514ea0>)), ('classifier', GaussianNB(priors=None))])
    	Accuracy: 0.80987	Precision: 0.30351	Recall: 0.32900	F1: 0.31574	F2: 0.32356
    	Total predictions: 15000	True positives:  658	False positives: 1510	False negatives: 1342	True negatives: 11490
    


>**Naive Bayes(Antes):** Accuracy: 0.84807	Precision: 0.38668	Recall: 0.23800	F1: 0.29465	F2: 0.25783

>**Naive Bayes(Depois):** Accuracy: 0.80987	Precision: 0.30351	Recall: 0.32900	F1: 0.31574	F2: 0.32356

### 5.3.2. Random Forest


```python
pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('reducer', PCA(random_state=42)),
        ('selector', SelectKBest()),
        ('classifier', RandomForestClassifier())
    ])
```


```python
param_grid = { 
    'classifier__n_estimators': [200, 700],
    'classifier__max_features': ['auto', 'sqrt', 'log2']
}
```


```python
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
```


```python
grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
```


```python
grid = grid_search.fit(features_train,labels_train)
```


```python
dump_classifier_and_data(grid_search.best_estimator_, my_dataset, features_list)
main()
```

    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reducer', PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('selector', SelectKBest(k=10, score_func=<function f_classif at 0x1a1f514ea0>)), ('classif...n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False))])
    	Accuracy: 0.85540	Precision: 0.40724	Recall: 0.18550	F1: 0.25490	F2: 0.20817
    	Total predictions: 15000	True positives:  371	False positives:  540	False negatives: 1629	True negatives: 12460
    


>**Random Forest(Antes):** Accuracy: 0.85467	Precision: 0.36527	Recall: 0.12200	F1: 0.18291	F2: 0.14075

>**Random Forest(Depois):** Accuracy: 0.85693	Precision: 0.41925	Recall: 0.18950	F1: 0.26102	F2: 0.21283

### 5.3.3. Decision Tree


```python
pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('reducer', PCA(random_state=42)),
        ('selector', SelectKBest()),
        ('classifier', tree.DecisionTreeClassifier())
    ])
```


```python
param_grid = {
    'classifier__criterion': ['gini','entropy'],
    'classifier__splitter': ['best', 'random'],
    'classifier__min_samples_split': [2,4,8,16],
    'classifier__class_weight': ['balanced', None],
    'classifier__min_samples_leaf': [1,2,4,8,16],
    'classifier__max_depth': [None,1,2,4,8,16],
}
```


```python
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
```


```python
grid_search = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
```


```python
grid = grid_search.fit(features_train,labels_train)
```


```python
dump_classifier_and_data(grid_search.best_estimator_, my_dataset, features_list)
main()
```

    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reducer', PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('selector', SelectKBest(k=10, score_func=<function f_classif at 0x1a1f514ea0>)), ('classif...      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'))])
    	Accuracy: 0.68027	Precision: 0.28670	Recall: 0.93950	F1: 0.43933	F2: 0.64553
    	Total predictions: 15000	True positives: 1879	False positives: 4675	False negatives:  121	True negatives: 8325
    


>**Decision Tree(Antes):** Accuracy: 0.79400	Precision: 0.23492	Recall: 0.24150	F1: 0.23817	F2: 0.24016

>**Decision Tree(Depois):** Accuracy: 0.68027	Precision: 0.28670	Recall: 0.93950	F1: 0.43933	F2: 0.64553

### 5.3.4. K-means


```python
pipe = Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('reducer', PCA(random_state=42)),
        ('selector', SelectKBest()),
        ('classifier', KMeans(n_clusters=2, random_state=0))
    ])
```


```python
param_grid = {
    'classifier__algorithm': ['auto','full','elkan'],
    'classifier__random_state': [42],
    'classifier__precompute_distances': ['auto', True, False],
    'classifier__max_iter': [10,50,100,200,400,500],
    'classifier__n_clusters': [2]
}
```


```python
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
```


```python
grid_search = GridSearchCV(pipe, param_grid, scoring='f1',cv=sss)
```


```python
grid = grid_search.fit(features_train,labels_train)
```


```python
dump_classifier_and_data(grid_search.best_estimator_, my_dataset, features_list)
main()
```

    Pipeline(memory=None,
         steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('reducer', PCA(copy=True, iterated_power='auto', n_components=None, random_state=42,
      svd_solver='auto', tol=0.0, whiten=False)), ('selector', SelectKBest(k=10, score_func=<function f_classif at 0x1a1f514ea0>)), ('classifier', KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=42, tol=0.0001, verbose=0))])
    	Accuracy: 0.72060	Precision: 0.14558	Recall: 0.22500	F1: 0.17678	F2: 0.20287
    	Total predictions: 15000	True positives:  450	False positives: 2641	False negatives: 1550	True negatives: 10359
    


>**K-means: Accuracy(Antes):** Accuracy: 0.85533	Precision: 0.31360	Recall: 0.07150	F1: 0.11645	F2: 0.084568

>**K-means: Accuracy(Depois):** Accuracy: 0.72060	Precision: 0.14558	Recall: 0.22500	F1: 0.17678	F2: 0.20287

### 5.3.5. Resultados Após Otimizações

>**Naive Bayes(Depois):** Accuracy: 0.80987	Precision: 0.30351	Recall: 0.32900	F1: 0.31574	F2: 0.32356

>**Random Forest(Depois):** Accuracy: 0.85693	Precision: 0.41925	Recall: 0.18950	F1: 0.26102	F2: 0.21283

>**Decision Tree(Depois):** Accuracy: 0.68027	Precision: 0.28670	Recall: 0.93950	F1: 0.43933	F2: 0.64553

>**K-means: Accuracy(Depois):** Accuracy: 0.72060	Precision: 0.14558	Recall: 0.22500	F1: 0.17678	F2: 0.20287

# 6. Conclusões

Neste caso de identificação de POI(Person of interest), ou seja, as pessoas que cometeram fraudes na emrpesa Eron, mas métricas mais significativas são a **precision** e a **recall**. 

O classificador que teve o melhor desempenho nos teste foi o **Decision Tree**, pois teve a métrica **F1** com maior valor. Essa métrica é a uma média harmonica entre precision e recall.

Os resultados do classificador são interpretados dessa forma:

- Recall 93.9%: Quando uma pessoa que é POI é submetida ao classificador, 93.9% das vezes será classificado como POI. em outras palavras, 6.1% de Falsos Negativos (Erro tipo II).
- Precision 28.6%: De todas as pessoas que são classificadas como POI, somente 28.6% são verdadeiros. Em outras palavras, temos 71.4% de Falsos positivos (Erro tipo I)
