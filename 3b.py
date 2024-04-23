import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
import random
from sklearn.preprocessing import LabelEncoder
# %matplotlib inline


class KNN:

    def __init__(self, nb_features, nb_classes, data, k, weighted=False):
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.k = k
        self.weighted = weighted
        self.X = tf.convert_to_tensor(data['x'], dtype=tf.float32)
        self.Y = tf.convert_to_tensor(data['y'], dtype=tf.int32)

    # Ako imamo odgovore za upit racunamo i accuracy.
    def predict(self, query_data):

        # Pokretanje na svih 10000 primera bi trajalo predugo,
        # pa pokrecemo samo prvih 100.
        nb_queries = len(query_data['x'])

        matches = 0
        for i in range(nb_queries):

            # Racunamo kvadriranu euklidsku udaljenost i uzimamo minimalnih k.
            dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, query_data['x'][i])), axis=1))
            _, idxs = tf.nn.top_k(-dists, self.k)

            classes = tf.gather(self.Y, idxs)
            dists = tf.gather(dists, idxs)

            if self.weighted:
                w = 1 / self.dists  # Paziti na deljenje sa nulom.
            else:
                w = tf.fill([self.k], 1/self.k)

            # Svaki red mnozimo svojim glasom i sabiramo glasove po kolonama.
            w_col = tf.reshape(w, (self.k, 1))
            classes_one_hot = tf.one_hot(classes, self.nb_classes)
            scores = tf.reduce_sum(w_col * classes_one_hot, axis=0)

            # Klasa sa najvise glasova je hipoteza.
            hyp = tf.argmax(scores)

            if query_data['y'] is not None:
                actual = query_data['y'][i]
                match = (hyp == actual)
                if match:
                    matches += 1
                # if i % 10 == 0:
                #     print(f'Test example: {i+1:2}/{nb_queries} | Predicted: {hyp} | Actual: {actual} | Match: {match}')

        accuracy = matches / nb_queries
        print(f'{matches} matches out of {nb_queries} examples')
        print(f'Test set accuracy: {accuracy}')
        return accuracy


if __name__ == "__main__":
    # Ucitavanje podataka iz fajla i shuffle-ovanje istih
    csvFile = pandas.read_csv("iris.csv", usecols=['sepal_length', 'sepal_width', 'species']).dropna()
    all_data = [[row[0], row[1], row[2]] for i, row in csvFile.iterrows()]
    random.shuffle(all_data)

    # Klase pretvaramo u numericke vrednosti
    label_encoder = LabelEncoder()
    all_labels = [row[2] for row in all_data]
    numeric_labels = label_encoder.fit_transform(all_labels)

    # Delimo podatke na trening i test skup
    split_index = int(0.85 * len(all_data))
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]

    # Odvajamo feature (x) i klase (y)
    train_X = [[row[0], row[1]] for row in train_data]
    train_Y = numeric_labels[:split_index]
    test_X = [[row[0], row[1]] for row in test_data]
    test_Y = numeric_labels[split_index:]

    # Treniramo KNN model za vrednosti k od 1 do 15, pamtimo accuracy
    k_vals = range(1, 16)
    res = []
    for k in k_vals:
        knn = KNN(nb_features=2, nb_classes=3, data={'x': train_X, 'y': train_Y}, k=k)
        res.append(knn.predict({'x': test_X, 'y': test_Y}))

    # Plotujemo
    plt.plot(k_vals, res, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('k-NN Accuracy for Different k')
    plt.show()

    # Kod izbora vrednosti za k, u slucaju kad je k malo, gleda se samo lokalna slika što može da dovede do
    # overfitting-a, dok sa druge strane ako je k veliko gleda se generalna slika i može da dovede do nepreciznosti
    # modela. U nasem slučaju za k = 7 je accuracy najoptimalniji