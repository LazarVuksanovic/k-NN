import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
import random

from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
# %matplotlib inline


def plot_data(train_X, train_Y):
    train_X = np.array(train_X, dtype=float)
    train_Y = np.array(train_Y, dtype=int)

    # Scatter plot
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    for label in np.unique(train_Y):
        idx = np.where(train_Y == label)
        points = train_X[idx]
        plt.scatter(points[:, 0], points[:, 1], label=label)

    # Contour plot
    x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
    y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    y_grid_pred = knn.predict_classes(X_grid).numpy()
    Z = y_grid_pred.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=cmap_light)

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Iris Dataset')
    plt.legend()
    plt.show()


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

    def predict_classes(knn, X):
        nb_queries = len(X)
        y_pred = []
        for i in range(nb_queries):
            # Slično kao u predict metodi, ali samo dodajemo predikcije u listu
            dists = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(knn.X, X[i:i + 1])), axis=1))
            _, idxs = tf.nn.top_k(-dists, knn.k)
            classes = tf.gather(knn.Y, idxs)
            votes = tf.reduce_sum(tf.one_hot(classes, depth=knn.nb_classes), axis=0)
            y_pred.append(tf.argmax(votes))
        return tf.stack(y_pred)


if __name__ == "__main__":
    # Ucitavanje podataka iz fajla i shuffle-ovanje istih
    csvFile = pandas.read_csv("iris.csv", usecols=['sepal_length', 'sepal_width', 'species']).dropna()
    all_data = [[row[0], row[1], row[2]] for i, row in csvFile.iterrows()]
    random.shuffle(all_data)

    # Klase pretvaramo u numericke vrednosti
    label_encoder = LabelEncoder()
    all_labels = [row[2] for row in all_data]
    numeric_labels = label_encoder.fit_transform(all_labels)

    # Delimo podatke na trening i test skup 70:30
    split_index = int(0.7 * len(all_data))
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]

    # Odvajamo feature (x) i klase (y)
    train_X = np.array([[row[0], row[1]] for row in train_data], dtype=float)
    train_Y = numeric_labels[:split_index]
    test_X = [[row[0], row[1]] for row in test_data]
    test_Y = numeric_labels[split_index:]

    # Treniramo KNN model
    knn = KNN(nb_features=2, nb_classes=3, data={'x': train_X, 'y': train_Y}, k=3)

    # Testiramo KNN model
    knn.predict({'x': test_X, 'y': test_Y})

    # Plotujemo
    plot_data(train_X, train_Y)