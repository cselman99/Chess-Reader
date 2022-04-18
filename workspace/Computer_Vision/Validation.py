# Validate Keras Model using cross validation method

def validateModel(model, trainX, trainY):
    validationRes = model.fit(x=trainX,
                              y=trainY,
                              validation_split=0.1,
                              batch_size=10,
                              epochs=5,
                              shuffle=True,
                              verbose=2)
    print(validationRes)
    return validationRes
