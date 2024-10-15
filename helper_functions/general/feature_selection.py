import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sm_discrete
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LogisticRegression

from helper_functions.afml_functions.afml_modelling_lib import PurgedKFold


def generate_heatmap(X):
    """
    Generates a heatmap for the given set of features (dataframe).

    Parameters:
    X (pd.DataFrame): DataFrame containing features for which the heatmap will be generated.

    Returns:
    None: Displays the heatmap.
    """
    # Calculate the correlation matrix
    corr_matrix = X.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Generate the heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)

    # Set plot labels and title
    plt.title('Feature Correlation Heatmap')

    # Show the heatmap
    plt.show()


def linear_model_with_pvalue_threshold(X, y, threshold=0.05, target_type='continuous'):
    """
    Fits a model (OLS/Logit/MNLogit) and drops features with p-values greater than the threshold.

    Parameters:
    X (pd.DataFrame): DataFrame containing the features.
    y (pd.Series): Target variable.
    threshold (float): The p-value threshold for feature selection. Default is 0.05.
    target_type (str): The type; continuous, binary, categorical - that the target variable takes.

    Returns:
    pd.DataFrame: DataFrame containing features with p-values below the threshold.
    dict: Dictionary of the p-values for each feature
    """
    # Initialise the model
    if target_type == 'continuous':
        fitter = sm.OLS
    elif target_type == 'categorical':
        fitter = sm_discrete.MNLogit
    elif target_type == 'binary':
        fitter = sm_discrete.Logit
    else:
        raise ValueError("Invalid target_type. Choose either 'categorical', 'binary' or 'continuous'.")

    # Add a constant to the features for the intercept
    X_with_const = sm.add_constant(X)

    # Fit the model
    model = fitter(y, X_with_const).fit()
    print(model.summary())

    # Get the p-values of the features
    p_values = model.pvalues

    # Select features with p-values below the threshold (excluding the constant term)
    significant_features = p_values[p_values <= threshold].index

    # Filter the original X to include only the significant features
    # We exclude the constant (intercept) term from the DataFrame we return
    X_significant = X_with_const[significant_features].drop(columns='const', errors='ignore')

    return X_significant, p_values


def linear_model_with_anova_f_test(X, y, threshold=0.05, boundary_threshold=0.1, target_type='continuous'):
    """
    Fits a model (OLS/Logit/MNLogit), removes boundary features with p-values close to the threshold,
    adds them back sequentially using ANOVA F-test to compare models, and retains
    significant features.

    Parameters:
    X (pd.DataFrame): DataFrame containing the features.
    y (pd.Series): Target variable.
    threshold (float): P-value threshold for significant features. Default is 0.05.
    boundary_threshold (float): Defines the range for boundary features. Default is 0.1.
    target_type (str): The type; continuous, binary, categorical - that the target variable takes.

    Returns:
    sm fitter object: The final model (OLS/Logit/MNLogit) with the significant features.
    pd.DataFrame: DataFrame containing the retained features.
    """
    # Initialise the model
    if target_type == 'continuous':
        fitter = sm.OLS
    elif target_type == 'categorical':
        fitter = sm_discrete.MNLogit
    elif target_type == 'binary':
        fitter = sm_discrete.Logit
    else:
        raise ValueError("Invalid target_type. Choose either 'categorical', 'binary' or 'continuous'.")

    # Step 1: Fit the initial model
    X_with_const = sm.add_constant(X)
    initial_model = fitter(y, X_with_const).fit()

    # Step 2: Get the p-values and identify boundary features
    p_values = initial_model.pvalues
    significant_features = p_values[p_values <= threshold].index
    boundary_features = p_values[(p_values > threshold) & (p_values <= boundary_threshold)].index

    # Remove constant from features to simplify boundary checking
    significant_features = significant_features.drop('const', errors='ignore')
    boundary_features = boundary_features.drop('const', errors='ignore')

    # Step 3: Fit the base model without the boundary features
    base_X = X[significant_features]
    base_model = fitter(y, sm.add_constant(base_X)).fit()
    print('base model', base_model.summary())

    # Step 4: Iterate over all combinations of boundary features
    best_r2 = base_model.rsquared_adj

    if len(boundary_features) > 0:
        for r in range(1, len(boundary_features) + 1):
            for combination in combinations(boundary_features, r):
                # Add this combination of boundary features to the base model
                test_X = pd.concat([base_X, X[list(combination)]], axis=1)
                test_model = fitter(y, sm.add_constant(test_X)).fit()

                # Perform ANOVA F-test between base model and test model
                anova_result = sm.stats.anova_lm(base_model, test_model)

                # If the test model is significantly better, keep the features
                if anova_result['Pr(>F)'][1] < threshold:
                    if test_model.rsquared_adj > best_r2:
                        best_r2 = test_model.rsquared_adj
                        significant_features = sm.add_constant(test_X)

    # Step 5: Refit final model with the selected features
    final_X = X[list(significant_features)]
    final_model = fitter(y, sm.add_constant(final_X)).fit()
    print('final model', final_model.summary())

    # Step 6: Plot the final model
    plt.figure(figsize=(10, 6))
    plt.scatter(y, final_model.fittedvalues, label="Fitted values", color='blue')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle="--", label="Perfect fit")
    plt.xlabel("Actual values")
    plt.ylabel("Fitted values")
    plt.title("Results")
    plt.legend()
    plt.show()

    return final_model, final_X


def feature_importance_mdi_pmf_with_rf(X, y, test_size=0.3, target_type='continuous'):
    """
    Performs feature importance analysis using Mean Decrease in Impurity (MDI) and
    Permutation Feature Importance (PFI) with a Random Forest model.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series or np.array): Target vector.
        test_size (float): The proportion of the dataset to include in the test split.
        target_type (str): 'categorical' for classification, 'continuous' for regression.

    Returns:
        pd.DataFrame: A DataFrame with the feature importance rankings from both MDI and PFI.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize the RandomForest model depending on the target_type
    if target_type == 'categorical':
        fitter = RandomForestClassifier(max_features=1, random_state=42)
    elif target_type == 'continuous':
        fitter = RandomForestRegressor(max_features=1, random_state=42)
    else:
        raise ValueError("Invalid target_type. Choose either 'categorical' or 'continuous'.")

    # Train the model
    fitter.fit(X_train, y_train)

    # Calculate feature importance using Mean Decrease in Impurity (MDI)
    mdi_importances = fitter.feature_importances_

    # Calculate feature importance using Permutation Feature Importance (PFI)
    pfi_result = permutation_importance(fitter, X_test, y_test, n_repeats=10, random_state=42)
    pfi_importances = pfi_result.importances_mean

    # Create a DataFrame to display the results side by side
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'MDI Importance': mdi_importances,
        'PFI Importance': pfi_importances
    })

    # Sort by MDI importance
    importance_df = importance_df.sort_values(by='MDI Importance', ascending=False)

    # Plot the feature importances for visualization
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Feature'], importance_df['MDI Importance'], color='blue', alpha=0.6, label='MDI Importance')
    plt.barh(importance_df['Feature'], importance_df['PFI Importance'], color='green', alpha=0.6, label='PFI Importance')
    plt.xlabel('Importance')
    plt.title('Feature Importance: MDI vs PFI')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

    # Return the importance DataFrame
    return importance_df


def create_pipeline_with_scaler_and_pca(explained_variance, X):
    """
    Creates a pipeline with StandardScaler and PCA, and returns the breakdown of original features into each component.

    Parameters:
    explained_variance (float): The amount of variance that PCA should retain (a value between 0 and 1).
    X (pd.DataFrame): The original dataset (before scaling).

    Returns:
    pipeline: A scikit-learn pipeline with StandardScaler and PCA.
    components_df: DataFrame showing the contributions of original features to each principal component.
    """

    # Ensure the explained variance is valid (between 0 and 1)
    if not 0 < explained_variance <= 1:
        raise ValueError("Explained variance must be between 0 and 1.")

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Scale the features
        ('pca', PCA(n_components=explained_variance))  # Step 2: Apply PCA with explained variance
    ])

    return pipeline.fit(X)


def get_pca_components(pipeline, X, feature_names=None):
    """
    Returns the breakdown of original features in terms of their contribution to each principal component.

    Parameters:
    pipeline: The fitted pipeline with StandardScaler and PCA.
    X (pd.DataFrame): The original dataset (before scaling).
    feature_names (list, optional): List of feature names. Defaults to the column names of X if X is a DataFrame.

    Returns:
    pd.DataFrame: DataFrame showing the feature contributions to each principal component.
    """
    # Fit the pipeline
    pipeline.fit(X)

    # Access the PCA object and its components
    pca = pipeline.named_steps['pca']

    # Use provided feature names or column names from X if it's a DataFrame
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]

    # Create a DataFrame showing the contribution of each feature to each principal component
    components_df = pd.DataFrame(pca.components_, columns=feature_names)
    components_df.index = [f'PC{i+1}' for i in range(len(components_df))]

    return components_df


def lasso_feature_selection(X, y, k=5, alpha_range=np.logspace(-4, 1, 50), target_type='continuous', shuffle=False, t1=None, pct_embargo=0.):
    """
    Perform feature selection using Lasso regression (or LogisticRegression for classification) with cross-validation.

    Parameters:
    X (pd.DataFrame or np.array): Feature matrix.
    y (pd.Series or np.array): Target variable.
    k (int): Number of folds for cross-validation.
    alpha_range (array-like): Range of alpha values to test.
    target_type (str): 'continuous' for Lasso, 'binary', or 'multinomial' for Logistic Regression.

    Returns:
    best_model: Fitted model with the best alpha.
    selected_features: List of feature names or indices with non-zero coefficients.
    """

    # Choose the correct estimator based on the target_type
    if target_type == 'continuous':
        model = Lasso(max_iter=10000)
    elif target_type == 'binary':
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    elif target_type == 'categorical':
        model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, multi_class='multinomial')
    else:
        raise ValueError("target_type must be 'regression', 'binary', or 'categorical'")

    cv_strategy = PurgedKFold(n_splits=k, t1=t1, pct_embargo=0., shuffle=shuffle, random_state=42)

    # Define the parameter grid (alpha values)
    param_grid = {'alpha': alpha_range} if target_type == 'regression' else {'C': 1/alpha_range}

    # Set up GridSearchCV to find the best alpha (or C for logistic regression)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy, scoring='neg_mean_squared_error' if target_type == 'regression' else 'neg_log_loss', n_jobs=-1)

    # Fit the grid search
    grid_search.fit(X, y)

    # Get the best model with the optimal alpha (or C)
    best_model = grid_search.best_estimator_

    # Fit the best model on the entire dataset
    best_model.fit(X, y)

    # Get the coefficients and identify the selected features
    if target_type == 'regression':
        coefficients = best_model.coef_
    else:  # TODO: Check whats going on here
        coefficients = best_model.coef_.ravel()  # For classification, take the raveled version of the coefficients

    # Get feature names if X is a DataFrame, otherwise use index
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    # Select features with non-zero coefficients
    selected_features = [feature_names[i] for i in range(len(coefficients)) if coefficients[i] != 0]

    return best_model, selected_features