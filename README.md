
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


# Kears參考手冊:https://keras.io

# 教學
以下是程式碼解析

結果顯示
![Imgur](https://i.imgur.com/6zrOQbd.gif)


## 1 執行
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
### 1-1mnist.load_data()
從keras.datasets取出mnist預設資料集，圖片如下：

![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

### 1-2np.expand_dims(X_train, axis=3)
在X_train中加上第三維度
- X_train:資料集
- axis: 維度位置

## 2 GAN物件
```
class GAN(object):
	def __init__(self, width=28, height=28, channels=1):
		....
		.....
    def __generator(self):
    	....
		.....
    def __discriminator(self):
    	....
		.....
    def __stacked_generator_discriminator(self):
    	....
		.....
    def train(self, X_train, epochs=100, batch = 32, save_interval = 100):
    	....
		.....
    def plot_images(self, save2file=False, samples=16, step=0):
    	....
		.....
```
### __init__
初始化設定

### __generator
生成器設定

### __discriminator
識別器設定

### __stacked_generator_discriminator
疊層生成&識別器設定

### train
進行訓練

### plot_images
用來生成圖片

## 初始化
進行初始設定，預先給定模型基礎設定
```python
def __init__(self, width=28, height=28, channels=1):
		#size:28 * 28 灰階

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
設定張量形狀，三個參數表示這是三階張量
- width:寬度
- height:長度
- channels:顏色通道

### optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
優化器設定，這邊採用Adam
- lr:float,學習速率
- beta_1:一階預估的衰減因子
- decay:學習後的衰減率參數

> Adam介紹:https://zhuanlan.zhihu.com/p/25473305

### self.G = self.__generator()
載入生成器模型設定

### self.D = self.__discriminator()
載入識別器模型設定

### self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
載入疊層生成&識別器設定

### compile(loss='binary_crossentropy', optimizer=self.optimizer,metrics=['accuracy'])
配置模型
- loss:設定損失函數，這邊採用binary_crossentropy，適合二元問題
- optimizer:配置優化器
- metrics:回傳指標數值，這邊採用accuracy

> 常用loss設定介紹:https://zhuanlan.zhihu.com/p/48078990

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

## 生成器和識別器交疊
固定discriminator 訓練generator
```python
def __stacked_generator_discriminator(self):

        self.D.trainable = False ##固定識別器參數

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model
```
## 訓練GAN
先訓練discriminator在訓練
```python
 def train(self, X_train, epochs=20000, batch = 32, save_interval = 100):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2)) 
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)   

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100)) 
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)
```

## 產生圖片
```python
def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
```
### plt
python的視覺化套件matplotlib

> matplotlib參考:https://ithelp.ithome.com.tw/articles/10186484

### noise = np.random.normal(0, 1, (samples, 100))
使用高斯分佈建立隨機樣本，回傳16組100個值的隨機生成陣列
- samples:取16個樣本數

### self.G.predict(noise)
交給生成器進行預測，回傳16組100個值的陣列

