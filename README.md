# deep-learning-implementations
My implementations of deep learning algorithms.

## autoencoder
My implementations of [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) in Tensorflow and Keras. Different variations of autoencoder are implemented, including Naive autoencoder, [denoising autoencoder](https://en.wikipedia.org/wiki/Autoencoder#Denoising_autoencoder), [sparse autoencoder](https://en.wikipedia.org/wiki/Autoencoder#Sparse_autoencoder), autoencoder with convolutional layer, [variational autoencoder](https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_.28VAE.29) (VAE) and conditional VAE.

#### [`autoencoder_tf.ipynb`](autoencoder/autoencoder_tf.ipynb)
Implementation of autoencoder in Tensorflow.

#### [`autoencoder_keras.ipynb`](autoencoder/autoencoder_keras.ipynb)
Implementation of autoencoder in Keras.


## GAN
My implementations of [generative adversarial networks](https://arxiv.org/pdf/1406.2661.pdf) (GAN) in Tensorflow. Different variations of GAN are implemented, including Naive GAN, GAN with convolutional layer, and conditional GAN.

#### [`GAN.ipynb`](GAN/GAN.ipynb)
Implementation of Naive GAN in Tensorflow.

#### [`GAN_conv.ipynb`](GAN/GAN_conv.ipynb)
Implementation of GAN with convolutional layer in Tensorflow.

#### [`conditional_GAN.ipynb`](GAN/conditional_GAN.ipynb)
Implementation of conditional GAN in Tensorflow.

# RNN
My implementations of [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) models.

#### [`imdb_rnn.py`](RNN/imdb_rnn.py)
Train a [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) model on the [IMDB sentiment classification dataset](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification).
