import AnayModel

testX = [[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 0]]
testY = [[1], [1], [1], [1], [0], [0], [0]]


mlp = AnayModel.mlp()
mlp.addInputLayer(3)
mlp.addDenseLayer(4, 'sigmoid')
mlp.addDenseLayer(1, 'sigmoid')
mlp.fit(testX, testY, 1000, learningRate=1, plot=True)
