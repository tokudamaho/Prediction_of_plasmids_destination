import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv
import sys
import os

def main(input_file, output_directory):
    # Load the CSV file into a pandas DataFrame
    # Assuming the last column contains the labels (y) and the rest are features (X)
    data = pd.read_csv(input_file, delimiter=",", skiprows=0, usecols=range(0, 16385))

    # Separate features (X) and labels (y)
    X = data.iloc[:, :-1]  # All columns except the last are features
    y = data.iloc[:, -1]   # The last column is the label

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]     # Minimum number of samples required to be at a leaf node
    }

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a RandomForestClassifier
    model = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and cross-validation score from the grid search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters found: ", best_params)
    print("Best cross-validation score: {:.2f}".format(best_score))

    # Train the best model using the entire training data
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", accuracy)

    # Generate a classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:", report)

    # Get the feature importances from the trained model
    feature_importances = best_model.feature_importances_
    for j, feature_name in enumerate(X.columns):
        print(f"{feature_name}: {feature_importances[j]}")

    # Save accuracy to a CSV file
    accuracy_file_path = os.path.join(output_directory, "Best_Accuracy.csv")
    with open(accuracy_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", accuracy])
    print("Accuracy results written to", accuracy_file_path)

    # Save classification report to a CSV file
    report_file_path = os.path.join(output_directory, "Best_Classification_Report.csv")
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(report_file_path)
    print("Classification report written to", report_file_path)

    # Save feature importances to a CSV file
    feature_importances_file_path = os.path.join(output_directory, "Best_Feature_Importances.csv")
    with open(feature_importances_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Feature", "Importance"])
        for j, feature_name in enumerate(X.columns):
            writer.writerow([feature_name, feature_importances[j]])
    print("Feature importances written to", feature_importances_file_path)

    # Save best parameters to a CSV file
    best_params_file_path = os.path.join(output_directory, "Best_Params.csv")
    with open(best_params_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for param, value in best_params.items():
            writer.writerow([param, value])
    print("Best parameters written to", best_params_file_path)

if __name__ == "__main__":
    # Check if the required command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_file output_directory")  # Usage instruction
        sys.exit(1)

    # Read command-line arguments
    input_file = sys.argv[1]  # Path to the input CSV file
    output_directory = sys.argv[2]  # Path to the directory for saving results

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Run the main function
    main(input_file, output_directory)
