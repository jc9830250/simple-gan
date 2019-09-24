
# Simple implementation of Adversarial Neural Network (GAN) using keras
Simple keras implementation of a generative adversarial neural network, as described in https://arxiv.org/abs/1406.2661

# How to run this repo
If you have virtualen installed run the following commands from the repo folder

```
virtualenv env

source env/bin/ativate

pip install -r requirements.txt

```
then you can run the code by simply: 
```
python gan.py

```

if you dont have virtual env installed you can install it like this: 

```
pip install virtualenv

```
# Medium article
see the companion article on Medium : https://medium.com/@mattiaspinelli/simple-generative-adversarial-network-gans-with-keras-1fe578e44a87



# 教學
以下是程式碼解析

首先是真正執行部分
```python
if __name__ == '__main__':
	(X_train, _), (_, _) = mnist.load_data()
	#取出預設的資料集圖像
	# Rescale -1 to 1
	X_train = (X_train.astype(np.float32) - 127.5) / 127.5
	X_train = np.expand_dims(X_train, axis=3)

	gan = GAN()
	gan.train(X_train)

```
### mnist.load_data()
從keras.datasets取出mnist預設資料集
![logo](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)


### np.expand_dims(X_train, axis=3)
在X_train中加上第三維度
- X_train:資料集
- axis: 維度位置


## 初始化
```python
def __init__(self, width=28, height=28, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


```

### shape(width,height,channels)
- width:
- height:
- channels:

### Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
- lr:
- beta_1:
- decay:

### compile(loss='binary_crossentropy', optimizer=self.optimizer)
- loss:
- optimizer:


## 生成器
```python
def __generator(self):
        """ Declare generator """

        model = Sequential()
        model.add(Dense(256, input_shape=(100,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
        model.add(Reshape((self.width, self.height, self.channels)))

        return model

```


## 識別器
```python
def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.int64((self.width * self.height * self.channels)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        return model
```




