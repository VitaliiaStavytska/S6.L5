
# Importo le librerie che mi servono

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import humanize as h


# Aggiungo il file csv da analizzare e lo stampo per controllare

try:
    insurance_data = pd.read_csv("insurance.csv")
except Exception as e:
    print(f"Errore durante il caricamento del file: {"insurance.csv"}")

print(insurance_data)

"""
Faccio l'analisi complesso del dadaset. Evidenzio i fattori principali che influiscono sul prezzo dell'assicurazione.
"""

# Vediamo prima i valori basilari

description_humanized = insurance_data.describe().applymap(lambda x: h.intword(x) if isinstance(x, (int, float)) else x)
print(description_humanized)

# Stampare i nomi delle colonne del dataframe

column_names = insurance_data.columns
print(column_names)

"""

Di se il dataset scelto rappresenta  un dataset della compagnia delle assicurazioni con 1338 clienti.
Il nostro dataset contiene 1338 righe e 7 colonne: "indice", "eta", "sesso", "BMI","quantita dei bambini","fumatore(si/no)", "regione" e "rata dell assicurazione".
Il quadro del cliente medio: eta di 39 anni, 1 bambino in carico, con BMI 30 e la rata media di 13270.42 euro. 
La rata minima venduta e 1.100 euro, invece quella massima e 63.800 euro.
L eta minima del cliente e 18 anni, massima di 64. 

"""

# Tasformo i dati stinghe in valori numerici per vedere la correlazione tra tutte le colonne

insurance_data['sex'] = insurance_data['sex'].map({'female': 0, 'male': 1})
insurance_data['smoker'] = insurance_data['smoker'].map({'yes': 1, 'no': 0})

# Utilizzo il one-hot encoding per la colonna 'region'

insurance_data1 = pd.get_dummies(insurance_data, columns=['region'], drop_first=True)

# Mappa dei codici alle regioni uniche

region_columns = [col for col in insurance_data1.columns if col.startswith('region_')]
region_mapping = dict(zip(insurance_data1[region_columns].columns, insurance_data1[region_columns].columns))

#Creo una matrice di correlazione di tutti i dati

correlation_matrix = insurance_data1.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matrice di Correlazione")
plt.show()

#Qui vediamo la correlazione generale tra tutte le  variabili presenti. Nei prossimi passi andiamo piu nel dettaglio

"""
Controlliamo la distribuzione delle variabili sui nostri acquirenti per avere uno quadro stabile del nostro segmento del mercato
"""

#Distribuzione di eta

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(insurance_data['age'], kde=True, color='skyblue')
plt.title('Distribuzione di eta')

# Piu grande quantita dei clienti si trova in range tra 18 e 20 anni di eta. Questa categoria dei clienti e prioritaria.
# Propongo di emettere i pacchetti-regalo per "Turning 18".
# Tutte le altre fasce di eta si distribuiscono quasi ugualmente. 

#Distribuzione di BMI

plt.subplot(2, 2, 2)
sns.histplot(insurance_data['bmi'], kde=True, color='salmon')
plt.title('Distribuzione di BMI')

# Obesita I grado: BMI 30- 34.9. Questa variabile appartiene alla magioranza dei nostri clienti. Si potrebbe provedere di 
# fornire copertura specifica per il diabete di tipo 2 o le malattie cardiache.

#Distribuzione della quantita dei bambini

plt.subplot(2, 2, 3)
sns.countplot(x='children', data=insurance_data, palette='pastel')
plt.title('Distribuzione della quantita dei bambini')

# Piu di 550 persone non hanno i bambini(fanno quasi la meta) del dataset intero(suppongo che e legato all eta giovane dei nostri acquirenti),
#  poco piu di 300 hanno solo 1 bambino, 200 ne hanno due, piu o meno 150 ne hanno 3 
# e pochissima percentuale ne hanno 4 e 5

#Distribuzione dei fumatori

plt.subplot(2, 2, 4)
sns.countplot(x='smoker', data=insurance_data, palette='pastel')
plt.title('Distribuzione dei fumatori')

# I fumatori fanno una gran parte nel dataset nostro, superano 1000 persone mentre non fumatori poco piu di 200. In totale sono 1338 persone.


# Distribuzione per il sesso

def plot_sex_distribution(data):
    
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sex', data=insurance_data, palette='pastel')
    plt.title('Distribuzione per il sesso')
    plt.tight_layout()
    plt.show()

plot_sex_distribution(insurance_data)

# Qui vediamo che la distribuzione del sesso tra i nostri acquirenti quasi pari a 50/50 con piccola prevalenza dei uomini



"""
Ora passiamo all analisi delle variabili principali che influiscono sul prezzo del nostro prodotto
"""

# Vediamo la dipendenza tra il sesso e il prezzo
plt.figure(figsize=(8, 5))
sns.boxplot(x='sex', y='charges', data=insurance_data, palette='Set2')
plt.title('Boxplot tra Charges e Sesso')
plt.xlabel('Sesso')
plt.ylabel('Costi (Charges)')
plt.show()

#Analizzando il grafico vediamo che il primo quartile e la mediana per uomini è le donne ha lo stesso valore. 
#Ma il terzo quartile (Q3) e l estensione verso il quarto quartile (superiore) sono molto piu pronunciati per gli uomini,
# significa che c e una maggiore variabilita nei dati degli uomini nella parte superiore della distribuzione.



# Regressione lineare tra 'age' e 'charges'
plt.figure(figsize=(8, 5))
sns.regplot(x='age', y='charges', data=insurance_data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title('Relazione tra Age e Charges con Regressione Lineare')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Il grafico di regressione che mostri un aumento graduale della tariffa di assicurazione all aumentare dell eta,
#  ma con un aumento abbastanza lento, suggerisce che l eta potrebbe avere un impatto sulla tariffa, ma non e il principale fattore determinante. 



# Vediamo la dipendenza tra la quantita dei figli e il prezzo

plt.figure(figsize=(8, 5))
sns.stripplot(x='children', y='charges', data=insurance_data, palette='Set3')
plt.title('Swarmplot tra il Numero di Figli e i Costi delle Assicurazioni')
plt.xlabel('Numero di Figli')
plt.ylabel('Costi (Charges)')
plt.show()

# Lo stripplot che mostra la relazione tra la quantita di bambini presi in carico e le tariffe delle assicurazioni suggerisce
#  che non c e una chiara correlazione lineare tra queste due variabili.


"""
Controlliamo quale reggione e piu pericolosa secondo della azienda assicurativa
"""

# Boxplot delle tariffe per regione

plt.figure(figsize=(8, 5))
sns.boxplot(data=insurance_data, x='region', y='charges', palette='Set1')
plt.title('Boxplot dei Costi per Regione')
plt.xlabel('Regione')
plt.ylabel('Costi (Charges)')
plt.show()

#Il boxplot delle tariffe per regione mostra chiaramente differenze significative nelle tariffe delle assicurazioni tra le diverse regioni. 
#In particolare, la regione "Southeast" sembra avere le tariffe più elevate rispetto alle altre regioni.
#Penso che la considerano come piu pericolosa da vivere dentro. 



# Scatterplot dei costi per BMI

plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=insurance_data, color='skyblue')
plt.title('Correlazione tra BMI e Charges')
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.show()

#Si osserva un aumento delle tariffe assicurative per i clienti con un BMI superiore a 30. 
#Questo e coerente con la definizione di obesita di primo grado (BMI 30-34.9), e suggerisce che vi potrebbe essere un aumento delle tariffe correlate all obesita.
#Variazione nelle tariffe: Nonostante l aumento di BMI, la variazione nelle tariffe non  uniforme. Solo una parte dei clienti con BMI superiore a 30 sembra avere tariffe 
#significativamente piu alte, mentre molti rimangono nella fascia di tariffe inferiore (1-2 mila euro).
#Conclusione preliminare: La correlazione tra BMI e tariffe sembra esistere, ma non e estremamente forte.
# Altri fattori potrebbero contribuire alla determinazione delle tariffe assicurative.


# Vediamo la distribuzione delle tariffe tra fumatori e non fumatori con un violin plot

plt.figure(figsize=(8, 5))
sns.violinplot(x='smoker', y='charges', data=insurance_data, palette='Set1')
plt.title('Violin Plot tra le Tariffe e Fumatori')
plt.xlabel('Fumatore')
plt.ylabel('Costi (Charges)')
plt.show()

#Il violin plot mostra chiaramente la differenza nella distribuzione delle tariffe tra fumatori e non fumatori. 
#Come abbiamo osservato, la maggior parte dei non fumatori ha tariffe inferiori a 2.000 euro, 
#mentre i fumatori mostrano una distribuzione piu ampia con tariffe che iniziano sopra i 2.000 euro.


"""
Ho individuato le variabili principali che influenzano i prezzi delle assicurazioni, come l eta, il fatto di essere fumatori o meno, 
il numero di figli, il BMI e la regione di residenza. Ora posso utilizzare queste informazioni per creare una tabella dei acquirenti di base, evitando i valori che hanno 
almeno piccolo impatto sul prezzo dell assicurazione.
"""

# prima di tutto dovevo trasferire le colonne in liste

age_list = insurance_data['age'].tolist()
sex_list = insurance_data['sex'].tolist()
bmi_list = insurance_data['bmi'].tolist()
children_list = insurance_data['children'].tolist()
smoker_list = insurance_data['smoker'].tolist()
region_list = insurance_data['region'].tolist()
charges_list = insurance_data['charges'].tolist()

"""
Nel prossimo passagio filtro i dati dei acquirenti in base alla correlazione.Ottenendo la sottotabella con i dati delle tariffe minime.
Prendiamo di base questi informazioni: non fumatori, donne, senza bambini, eta sotto di 30 anni, BMI tra 18,5 e 24,9 come dalla norma(presa su google), regione che non e southeast.
"""

def filter_data(age_list, smoker_list, children_list, charges_list, bmi_list, region_list):
    # Creazione di un DataFrame dai dati delle liste
    df = pd.DataFrame({
        'age': age_list,
        'smoker': smoker_list,
        'children': children_list,
        'charges': charges_list,
        'bmi': bmi_list,
        'region': region_list,
         'sex': sex_list
    })

    # Filtraggio del DataFrame secondo i criteri
    filtered_df = df[(df['smoker'] == 0) & (df['children'] == 0) & (df['age'] < 30) & (df['bmi'] >= 18.5) & (df['bmi'] <= 24.9) & (df['region'] != "southeast") & (df['sex'] != 1)]

    return filtered_df

# Chiamo della funzione con le liste
filtered_df = filter_data(age_list, smoker_list, children_list, charges_list, bmi_list, region_list)

# Applico la formattazione leggibile agli elementi numerici del DataFrame
filtered_df_humanized = filtered_df.applymap(lambda x: h.intword(x) if isinstance(x, (int, float)) else x)

# Stampa del DataFrame risultante con valori umanizzati
print(filtered_df_humanized)

# Analisi di base del DataFrame ottenuto con valori umanizzati
description_humanized = filtered_df.describe().applymap(lambda x: h.intword(x) if isinstance(x, (int, float)) else x)
print(description_humanized)

#A questa descrizione corrispondono 15 acquirenti. La loro tariffa media e 2.600 euro, la BMI media e 22, e loro si trovano in range tra 19 e 27 anni.
#Tutto questo mi fa pensare che ci sono altri fattori che influiscono sul prezzo ma non sono presenti nel dataset.


"""

Basandosi sull analisi svolta, ecco alcuni consigli e possibili azioni per compania assicurativa:

Pacchetti per il Gruppo di Eta 18-20 Anni: considerando la concentrazione di clienti tra 18 e 20 anni, si potrebbe sviluppare pacchetti speciali o promozioni mirate a questa
fascia di eta. Ad esempio, offerte legate a eventi come "Turning 18" potrebbero attirare l attenzione.

Offerte per Clienti Non Fumatori:data la differenza significativa nelle tariffe tra fumatori e non fumatori, si potrebbe creare pacchetti assicurativi speciali o sconti per
i clienti non fumatori al fine di incentivare comportamenti salutari.

Analisi Approfondita sulle Regioni: poiche la regione "Southeast" sembra avere tariffe piu elevate, potrebbe essere vantaggioso condurre un analisi piu approfondita sui 
fattori specifici che contribuiscono a questo aumento di costo. Cio potrebbe portare a strategie di prezzo piu mirate per quella regione.

Esplorazione di Altri Fattori: riguardo altri fattori che potrebbero influire sui prezzi ma non sono presenti nel dataset, si potrebbe cercare dati aggiuntivi per una
comprensione piu completa.

Personalizzazione delle Offerte: si considera l implementazione di strategie di personalizzazione delle offerte. Ad esempio, offerte speciali per clienti con BMI piu elevato o 
pacchetti specifici per coloro con figli potrebbero essere ben accolti.
"""


