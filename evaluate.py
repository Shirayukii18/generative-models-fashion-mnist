"""Evaluasi checkpoint yang sudah lengkap."""
from src.pipeline import prepare_data, train_all
from src.evaluation import evaluate, create_figures

if __name__ == '__main__':
    data = prepare_data()
    models = train_all(data)
    metrics, summary = evaluate(models, data)
    create_figures(models, data)
    print(metrics.to_string(index=False))
