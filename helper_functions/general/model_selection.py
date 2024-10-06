import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV


def randomized_grid_search(pipeline, param_distributions, cv, X, y, n_iter=50, scoring='accuracy', random_state=None):
    """
    Performs a RandomizedSearchCV on a given pipeline and displays model performance.

    Args:
    pipeline: scikit-learn pipeline or estimator object.
    param_distributions: Dictionary with hyperparameters to search over.
    cv: Cross-validation splitting strategy.
    X: Features dataset.
    y: Target dataset.
    n_iter: Number of parameter settings sampled (default is 50).
    scoring: Scoring metric for model evaluation (default is 'accuracy').
    random_state: Random state for reproducibility.

    Returns:
    Best model from the search.
    """
    # Create the RandomizedSearchCV object
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        return_train_score=True,
        n_jobs=-1, # Use all processors for efficiency
        verbose=1
    )

    # Fit the search object
    search.fit(X, y)

    # Results as a pandas DataFrame
    results = pd.DataFrame(search.cv_results_)

    # Display top 10 hyperparameter combinations and their scores
    top_results = results.nlargest(10, 'mean_test_score')

    print("\nTop 10 Hyperparameter Combinations (by Mean Test Score):")
    print(top_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    # Display graphical representation of top models
    fig, ax = plt.subplots(figsize=(10, 6))
    top_results.plot(
        x='rank_test_score',
        y='mean_test_score',
        kind='bar',
        yerr='std_test_score',
        ax=ax,
        color='skyblue',
        capsize=4
    )
    ax.set_title('Top 10 Models Performance')
    ax.set_xlabel('Rank of Model (Lower is Better)')
    ax.set_ylabel('Mean Test Accuracy Score')
    ax.set_xticklabels(top_results['rank_test_score'].astype(int), rotation=0)
    plt.show()

    # Display the best model and hyperparameters
    print("\nBest Hyperparameters found:")
    print(search.best_params_)

    print(f"\nBest Cross-Validation Score: {search.best_score_:.4f}")

    return search.best_estimator_
