import numpy as np
from collections import Counter
from scipy.spatial import distance
from random import seed
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats

# Load the dataset from a file and preprocess it
def load_and_preprocess_dataset(filename):
    dataset = []

    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            buying, maint, doors, persons, lug_boot, safety, class_label = data

            # Convert doors and persons to numerical values
            doors = 4 if doors == '5more' else int(doors)
            persons = 4 if persons == 'more' else int(persons)

            # Use a dictionary to map string values to numerical values for the other attributes
            buying_map = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
            maint_map = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
            lug_boot_map = {'small': 1, 'med': 2, 'big': 3}
            safety_map = {'low': 1, 'med': 2, 'high': 3}
            class_map = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

            # Encode categorical attributes
            buying_encoded = buying_map[buying]
            maint_encoded = maint_map[maint]
            lug_boot_encoded = lug_boot_map[lug_boot]
            safety_encoded = safety_map[safety]
            class_encoded = class_map[class_label]

            # Combine all values
            encoded_data = [buying_encoded, maint_encoded, doors, persons, lug_boot_encoded, safety_encoded, class_encoded]
            dataset.append(encoded_data)

    return dataset

# Calculate the Euclidean distance between two instances
def euclidean_distance(instance1, instance2):
    return distance.euclidean(instance1[:-1], instance2[:-1])

# Make predictions using KNN
def knn_predict(train_data, test_instance, k):
    distances = [(data, euclidean_distance(test_instance, data)) for data in train_data]
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    neighbor_labels = [neighbor[0][-1] for neighbor in neighbors]
    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    return prediction

# Perform K-Fold Cross Validation
def k_fold_cross_validation(dataset, k, k_value):
    np.random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    accuracies = []

    for i in range(k):
        test_set = folds[i]
        train_set = np.vstack([folds[j] for j in range(k) if j != i])

        correct = 0
        for test_instance in test_set:
            prediction = knn_predict(train_set, test_instance, k_value)
            if prediction == test_instance[-1]:
                correct += 1

        accuracy = correct / len(test_set) * 100
        accuracies.append(accuracy)

    return accuracies

# Hyperparameter Tuning - Grid Search
def grid_search(dataset, k_fold, k_values):
    best_k_value = None
    best_accuracy = 0

    for k_value in k_values:
        custom_knn_accuracies = k_fold_cross_validation(dataset, k_fold, k_value)
        mean_accuracy = np.mean(custom_knn_accuracies)

        if mean_accuracy > best_accuracy:
            best_k_value = k_value
            best_accuracy = mean_accuracy

    return best_k_value

# Hypothesis Testing - Paired T-Test
def paired_t_test(custom_knn_accuracies, sklearn_knn_accuracies):
    t_stat, p_value = stats.ttest_rel(custom_knn_accuracies, sklearn_knn_accuracies)
    return t_stat, p_value

if __name__ == "__main__":
    seed(1)

    training_data = load_and_preprocess_dataset('car.data')

    # Number of neighbors(K) values for GridSearchCV
    k_values = [1, 3, 5, 7, 9]

    # Hyperparameter tuning to find the best K value
    best_k_value = grid_search(training_data, k_fold=10, k_values=k_values)

    print("Best K value:", best_k_value)

    # Use the best K value for K-Fold Cross Validation for custom KNN implementation
    custom_knn_accuracies = k_fold_cross_validation(training_data, 10, best_k_value)

    # Use scikit-learn's KNN for comparison
    features = [data[:-1] for data in training_data]
    labels = [data[-1] for data in training_data]
    knn = KNeighborsClassifier(n_neighbors=best_k_value)
    sklearn_knn_accuracies = cross_val_score(knn, features, labels, cv=10)

    # Output the results
    print("Custom KNN Accuracies:", custom_knn_accuracies)
    print("Scikit-learn KNN Accuracies:", sklearn_knn_accuracies)

    # Hypothesis testing
    t_stat, p_value = paired_t_test(custom_knn_accuracies, sklearn_knn_accuracies)
    print("Paired T-Test - t-statistic:", t_stat)
    print("Paired T-Test - p-value:", p_value)
