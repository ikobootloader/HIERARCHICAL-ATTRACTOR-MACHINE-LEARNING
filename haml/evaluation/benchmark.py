"""
Module benchmark.

Compare HAML avec des baselines classiques.
"""

import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def run_benchmark(haml_model, X_train, y_train, X_test, y_test, baselines=['knn', 'svm']):
    """
    Benchmark HAML contre des baselines.

    Args:
        haml_model: Modèle HAML (déjà entraîné)
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        baselines (list): Modèles de comparaison

    Returns:
        dict: Résultats {model_name: {accuracy, time}}
    """
    results = {}

    # HAML
    start = time.time()
    y_pred_haml = haml_model.predict(X_test)
    time_haml = time.time() - start

    acc_haml = np.mean(y_pred_haml == y_test)
    results['HAML'] = {'accuracy': acc_haml, 'time': time_haml}

    print(f"HAML: Accuracy = {acc_haml:.3f}, Time = {time_haml:.3f}s")

    # Baselines
    if 'knn' in baselines:
        knn = KNeighborsClassifier(n_neighbors=5)
        start = time.time()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        time_knn = time.time() - start

        acc_knn = np.mean(y_pred_knn == y_test)
        results['KNN'] = {'accuracy': acc_knn, 'time': time_knn}
        print(f"KNN: Accuracy = {acc_knn:.3f}, Time = {time_knn:.3f}s")

    if 'svm' in baselines:
        svm = SVC(kernel='rbf')
        start = time.time()
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        time_svm = time.time() - start

        acc_svm = np.mean(y_pred_svm == y_test)
        results['SVM'] = {'accuracy': acc_svm, 'time': time_svm}
        print(f"SVM: Accuracy = {acc_svm:.3f}, Time = {time_svm:.3f}s")

    return results
