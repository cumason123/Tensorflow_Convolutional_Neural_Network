from ConvolutionalNeuralNetwork import *

if __name__ == "__main__":
    if not os.path.exists("funcDirectory/"):
        GenData.gendata()
    # make_test_data(np.arcsinh)
    nn = CryptoNeuralNetwork()
    nn.train_model(iterations=2000, load_checkpoint=False)
    #results = nn.make_prediction(cv2.imread("testDirectory/arcsinh.jpeg"))

