import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df):
    metrics = ['Accuracy', 'F1 Score', 'Recall']
    results_df.set_index("Model")[metrics].plot(kind='bar', figsize=(10, 6))
    plt.title("Model Comparison — Accuracy, F1 Score, Recall")
    plt.ylabel("Score")
    plt.show()
