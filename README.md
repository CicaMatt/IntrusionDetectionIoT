# Repository della tesi "Implementazione di un Sistema di Intrusion Detection per l'IoT tramite Machine Learning"

## Download Dataset
Il dataset oggetto di studio utilizzato é scaricabile presso il seguente link: https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
In alternativa, é possibile scaricarlo utilizzando l'apposita funzione nel file dataset_download.py, previa modifica del file kaggle.json con gli appositi campi username e key.

## Descrizione Dataset
Il dataset, in formato .csv, contiene informazioni relative ad un traffico di dati reale, ottenuto da 9 diversi dispositivi IoT di natura commerciale, infettati in modo autentico da malware della famiglia Mirai e BASHLITE. Il dataset conta 7062606 record, ad ognuno dei quali sono associate 115 features. É presente una suddivisione in 9 sottogruppi che corrispondono ai 9 dispositivi esaminati, e per ognuno di essi é presente un ulteriore partizionamento effettuato in relazione all’attacco a cui sono soggetti. Oltre ai dati sul traffico benigno, sono presenti dati relativi al traffico anomalo, che possono essere divisi in 10 sottogruppi relativi rispettivamente a 10 attacchi portati avanti da reti botnet della famiglia Mirai e BASHLITE, dunque si presta bene per una classificazione multi-classe, per un totale di 10 classi di attacchi più una classe di traffico normale (tali classi sono suddivise sottoforma di file .csv).
Le relazioni fra le feature sono mostrate nei due grafici presenti nella repository, e sono generabili sul proprio dispositivo eseguendo il file data_exploration.py.

## Descrizione Pipeline
Per rendere possibili il confronto e rendere più atomici i risultati ottenuti dal modello, ogni run effettuerà la classificazione selettivamente per il dispositivo selezionato, dunque saranno necessarie 9 run per ottenere informazioni su tutti i dispositivi. 
Per la lettura dei file .csv e le successive operazioni é stata utilizzata la libreria pandas, con la quale vengono letti gli 11 file .csv (1 traffico benigno + 10 traffico anomalo) e trasformati in formato DataFrame, in modo da poter trattare i dati sotto forma di tabelle bidimensionali. 
Per quanto riguarda i dispositivi Philips e Samsung, essi mancano di dati relativi ad un'infezioni Mirai, pertanto l'analisi di questa tipologia di dati é omessa.
Successivamente, per ognuna delle tabelle ottenute viene effettuato un campionamento randomico con lo scopo di ottenere un numero di campioni uguale per ognuna delle classi di malware da predire. In questo modo, si evita che in seguito siano presenti un numero di osservazioni diversi per ogni classe, e di conseguenza vengono evitati dal principio fenomeni di oversampling e undersampling.
Infine, ad ognuna delle tabelle viene aggiunta la colonna ’type’, a cui viene associata una stringa che permette di identificare la classe di appartenenza della tabella in questione, per poi fonderle insieme in un’unica tabella, che costituirà in toto l’effettivo dataset a partire dal quale verranno effettuate tutte le operazioni di validazione, preprocessing e classificazione.

Ogni classificatore presenta la propria classe. Come prima operazione, viene scelto il dispositivo per il quale saranno esaminati i dati.
Successivamente avviene in ordine la rimozione dei duplicati e lo shuffling delle righe, per ridurre la possibilitá di anomalie durante l'addestramento.
Dopodiché, viene creato un clone del dataset in formato Dataframe prendendo solo la colonna 'type' che contiene la classe di appartenza di ogni entry, che sará utilizzato per la fase di prediction, per poi eliminare la suddetta colonna in modo da poter utilizzare il DataFrame originale per l'addestramento.

Troviamo poi le tecniche di normalizzazione, fra cui Standard Scaling e Min-Max Normalization.
Di seguito, sono poi presenti le varie tecniche di preprocessing, esposte in ordine:
- Rimozione di outlier con approccio Z-Score
- Rimozione di outlier con approccio Interquartile Range
- Feature Selection, con gli approcci di Soglia di Varianza e Selezione delle K migliori features
- Feature Extraction con approccio PCA
Le tecniche citate sono utilizzabili sia singolarmente, sia opportunamente in modo combinato.

Troviamo poi la fase di validazione tramite train/test split, dove avviene lo splitting 80/20 randomico.
Successivamente c'é la fase di addestramento vera e propria di ogni classificatori, con i parametri ottimali giá configurati, seguita dalla fase di prediction sfruttando i dati ottenuti tramite la clonazione precedente.

Infine, per quanto riguarda le metriche, la loro produzione é racchiusa nella classe metrics.py, che effettua la stampa delle metriche di accuracy, precision, recall, F1 score, insieme a matrici di confusioni binarie e multiclasse.




