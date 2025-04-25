from sklearn.utils.class_weight import compute_class_weight

def train_model(model, train, val, y_train, epochs=10):
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Train model
    history = model.fit(train, validation_data=val, epochs=epochs, class_weight=class_weights_dict)

    return history
