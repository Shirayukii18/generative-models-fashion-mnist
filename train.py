"""Jalankan training dari terminal atau notebook."""
from src.pipeline import prepare_data, train_all

if __name__ == '__main__':
    data = prepare_data()
    models = train_all(data)
