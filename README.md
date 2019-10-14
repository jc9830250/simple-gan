
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

結果顯示
![Imgur](https://i.imgur.com/YOp9mKP.gif)


## 1 執行
```
plt.switch_backend('agg')   # allows code to run without a system DISPLAY
```
cmd指令可以繪製圖案

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

### X_train = (X_train.astype(np.float32) - 127.5) / 127.5
將圖像資料做正規化，縮限在-1到1之間
RGB最大值為255，在這邊資料都為灰階

### 1-2np.expand_dims(X_train, axis=3)
在X_train中的第四個位置加上資料
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
### __init__:初始化設定

### __generator:生成器設定

### __discriminator:識別器設定

### __stacked_generator_discriminator:疊層生成&識別器設定

### train:進行訓練

### plot_images:用來生成圖片

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

> Adam介紹: 
> - https://zhuanlan.zhihu.com/p/25473305
> - https://keras.io/zh/optimizers/
https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7

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
### LeakyReLU(alpha=0.2)
- alpha: 斜率
- LeakyReLU避免神經元死亡，就算沒激活仍然還是有一定斜率
> 可參考:
> - https://keras.io/zh/layers/normalization/
> - http://sofasofa.io/forum_main_post.php?postid=1001234
> - https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/

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
> 可參考:
> - https://ithelp.ithome.com.tw/articles/10204106
> - http://violin-tao.blogspot.com/2018/02/ml-batch-normalization.html
> - https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/

### tanh()
- Hyperbolic tangent function
- 激活函數的一種
> 可參考:https://ithelp.ithome.com.tw/articles/10189085

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
### sigmoid
-輸出範圍介於[0, 1]
-用來判斷圖片的真假程度
-可參考:
- https://ithelp.ithome.com.tw/articles/10189085

## 層生成&識別器
固定discriminator 訓練generator
在這邊會綁定兩個模型，讓discriminator帶著generator一起訓練
但不更動discriminator的模型
```python
def __stacked_generator_discriminator(self):

        self.D.trainable = False ##固定識別器參數

        model = Sequential()
        
        #將generator和discriminator合併
        model.add(self.G) 
        model.add(self.D)
        return model
```
### self.D.trainable = False 
固定識別器，只讀取參數不做影響

## 進行訓練
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
### gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100)) 
隨機高斯產生分部的數值 均值為0，標準差為1，產生16個陣列，裡面包含100個數值

> numpy.random.normal:https://blog.csdn.net/lanchunhui/article/details/50163669

### syntetic_images = self.G.predict(gen_noise)
利用生成器產生假圖片，給discriminator訓練用的假圖片(self.width, self.height, self.channels)

### x_combined_batch = np.concatenate((legit_images, syntetic_images)) 
合併真假圖片樣本

### y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1)))) 
合併標籤，真圖片分類給1，假圖片給0


## 圖片生成
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
- figure:設定參數
- subplot:在一個畫布上給予子圖
- imshow:顯示圖片
- axis:顯示軸線
- tight_layout:填充子圖至整個畫布
- savefig:儲存圖像
- close:關閉畫布
- show:顯示圖像

> matplotlib參考:https://ithelp.ithome.com.tw/articles/10186484

### noise = np.random.normal(0, 1, (samples, 100))
使用高斯分佈建立隨機樣本，回傳16組100個值的隨機生成陣列
- samples:取16個樣本數

### self.G.predict(noise)
交給生成器進行預測，回傳16組100個值的陣列

## 參考資料
- Kears參考手冊:
https://keras.io

- LeakyReLU介紹:
https://keras-cn.readthedocs.io/en/latest/layers/advanced_activation_layer/

- NumPy用法:
https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html

- Keras Sequential 顺序模型:
https://keras.io/zh/getting-started/sequential-model-guide/

- keras Flatten:
https://blog.csdn.net/qq_33221533/article/details/82256531

- numpy.reshape:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

-核心网络层
https://keras.io/zh/layers/core/

-activation function
https://ithelp.ithome.com.tw/articles/10189085

-張量
https://www.slideshare.net/ccckmit/ss-102849756

