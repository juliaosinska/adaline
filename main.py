from data import load_data, prepare_ova_labels
from train import train_models, predict, evaluate
from plot import plot_errors

def main():
    # data loading
    x_train, y_train, x_test, y_test = load_data()
    Y_train = prepare_ova_labels(y_train)

    # training
    models = train_models(x_train, Y_train, epochs=50)

    # test
    y_pred = predict(models, x_test)
    acc, err = evaluate(y_pred, y_test)

    print(f"\nDokładność: {acc:.4f}")
    print(f"Błąd: {err:.4f}")

    # plotting error curves
    plot_errors(models)

if __name__ == "__main__":
    main()