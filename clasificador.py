import pandas as pd
import nltk
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


data = pd.read_csv("corpus_spam.csv", header = None, names = ["Spam", "Mensaje"])
stemmer = PorterStemmer()


#Esta parte del código es el procesamiento de los datos, ya que convertimos todos en minisculas y limpiamos cualquier puntuación
data['Mensaje'] = data.Mensaje.map(lambda x: x.lower())
data['Mensaje'] = data.Mensaje.str.replace('[^ws]', '')
print(data)


# Utilizamos nltk para convertir los mensajes en una sola palalabra
nltk.download('punkt')
data['Mensaje'] = data['Mensaje'].apply(nltk.word_tokenize)


#Se realiza una derivación de palabras, para que no importe el tiempo verbal
data['Mensaje'] = data['Mensaje'].apply(lambda x: [stemmer.stem(y) for y in x])


# Esto convierte la lista de palabras en cadenas separadas por espacios
data['Mensaje'] = data['Mensaje'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(data['Mensaje'])
transformer = TfidfTransformer().fit(counts)
counts = transformer.transform(counts)


# Entrenamos al modelo
X_train, X_test, y_train, y_test = train_test_split(counts, data['Spam'], test_size=0.1, random_state=120)
model = MultinomialNB().fit(X_train, y_train)


#Evaluación del modelo con la librería de numpy
predicted = model.predict(X_test)

print(np.mean(predicted == y_test))


#Confusion Matrix
#			Predicted No 	Predicted Yes
#Actual No
#Actual Yes
print( confusion_matrix(y_test, predicted) )