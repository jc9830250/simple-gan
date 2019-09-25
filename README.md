
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

## 1執行
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

## GAN物件
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



## 初始化
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
### LeakyReLU(alpha=0.2)
- alpha: 斜率
- LeakyReLU避免神經元死亡，就算沒激活仍然還是有一定斜率
-可參考:
https://keras.io/zh/layers/normalization/
http://sofasofa.io/forum_main_post.php?postid=1001234
https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/

### Dense()
- 全連結層
- units: 輸出維度 ex:256。
- input_shape: 輸入的維度大小 這邊100維
- activation: 所使用的激活函數
- Reshape: 調整輸出的尺寸 ex(None,28,28,1)

### BatchNormalization()
- 作標準化，加速神經網路
- 減去平均除以標準差
- momentum 動量
- 可參考:
 -- https://ithelp.ithome.com.tw/articles/10204106
 -- http://violin-tao.blogspot.com/2018/02/ml-batch-normalization.html
 -- https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/

### tanh()
- Hyperbolic tangent function
- 激活函數的一種
- 可參考:
 https://ithelp.ithome.com.tw/articles/10189085

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
## sigmoid
-輸出範圍介於[0, 1]
用來判斷圖片的真假程度
可參考:
https://ithelp.ithome.com.tw/articles/10189085

## 生成器和識別器交疊

固定discriminator 訓練generator
```python
def __stacked_generator_discriminator(self):

        self.D.trainable = False ##固定識別器參數

        model = Sequential()
	#將generator和discriminator合併
        model.add(self.G) 
        model.add(self.D)
        return model
```


## GAN訓練
批次訓練，先訓練discriminator，再訓練generator
```python
 def train(self, X_train, epochs=20000, batch = 32, save_interval = 100):

        for cnt in range(epochs): #訓練20000次

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2)) #隨機產生index
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)   #從raw data隨機抽取真實圖片資料
		#隨機高斯產生分部的數值 均值為0，標準差為1，產生16個陣列，裡面包含100個數值
            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100)) 
            syntetic_images = self.G.predict(gen_noise) #利用生成generator假圖片，給discriminator訓練用的假圖片(self.width, self.height, self.channels)

            x_combined_batch = np.concatenate((legit_images, syntetic_images)) #合併真假圖片
	    #真圖片分類給1，假圖片給0，合併
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1)))) 

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch) #訓練discriminator分辨真假圖片


            # train generator

            noise = np.random.normal(0, 1, (batch, 100)) #隨機生成高斯分部數值，32個陣列，裡面包含100個數值
            y_mislabled = np.ones((batch, 1)) #產生100個為1的陣列，將上面數值所產生當作真圖片

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled) #fix discriminator 訓練generator
	    #d_loss代表Discriminator cross entropy loss value，值越小代表分類器分的越好
	    #g_loss代表在fix Discriminator情況下， Discriminator分辨generator產生圖片的cross entropy loss value，值越小代表generator越好
	    print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
            if cnt % save_interval == 0: #每100次生成手寫圖片
                self.plot_images(save2file=True, step=cnt)
```
## 參考資料
LeakyReLU介紹
https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/

NumPy用法
https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html

Keras Sequential 顺序模型
https://keras.io/zh/getting-started/sequential-model-guide/

keras Flatten
https://blog.csdn.net/qq_33221533/article/details/82256531

numpy.reshape
https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

Adam
https://keras.io/zh/optimizers/
https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7

numpy.random.normal
https://blog.csdn.net/lanchunhui/article/details/50163669

核心网络层
https://keras.io/zh/layers/core/

activation function
https://ithelp.ithome.com.tw/articles/10189085
