import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_utils import load_datasets
from src.model_utils import build_model
from src.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train plant disease MobileNetV2 classifier")
    parser.add_argument("--data_dir", default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune", action="store_true")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    model = build_model(len(class_names), train_base=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", monitor="val_accuracy",
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if args.fine_tune:
        backbone = model.get_layer("mobilenetv2_1.00_224")
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=max(2, args.epochs//2), callbacks=callbacks)

    best_path = Path("models/best_model.keras")
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save("models/plant_disease_mobilenetv2.keras")
    Path("models/class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    report = evaluate_model(model, test_ds, class_names)
    print("\nTest classification report:\n", report)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history.get("accuracy", []), label="train accuracy")
    ax.plot(history.history.get("val_accuracy", []), label="validation accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(True, alpha=.25)
    fig.tight_layout(); fig.savefig("outputs/training_history.png", dpi=160); plt.close(fig)
    print("Saved model to models/plant_disease_mobilenetv2.keras")

if __name__ == "__main__":
    main()
