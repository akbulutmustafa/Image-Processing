#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np

class Perceptron(object):
    
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1) # 0. satıra 1
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
                


# In[80]:


X = np.array([
       [0, 0],
       [0, 1],
       [1, 0],
       [1, 1]
   ])


# In[65]:


d = np.array([0, 0, 0, 1])
d


# In[75]:


perceptron.W


# In[76]:


perceptron = Perceptron(input_size=2)


# In[77]:


perceptron.fit(X, d)


# In[78]:


print(perceptron.W)


# In[41]:


mp = Perceptron(5)
x = np.asarray([-10,-2,-3,-40,-5])
mp.predict(x)


# In[44]:


mp.W


# init fonksiyonunda girdi sayısını, döngü sayısını ve öğrenme sıklığını tanımladık.
# girdi sayısını 1 fazla almamızın sebebi bias için yer ayırmak
# 
# activation_fn de eğer x 0'dan büyük veya eşit ise 0, küçük ise 1 döndürüyoruz
# 
# predict fonksiyonununda öncelikle x'e bias'ı ekliyoruz 
# ağırlıklar ve x değerlerini çarpıyoruz ve sonucu aktivasyon fonksiyonuna gönderip gelen değeri dönüyoruz
# 
# fit fonksiyonunda ise tek tek her x elemanı için predict yapılarak y değerleri elde edilir ve beklenen değerden y değeri 
# çıkarılarak hata hesaplanır. sonrasında hata değeri, öğrenme sıklığı ve x'in o anki elemanı çarpılıp eski ağırlıklarla toplandıktan 
# sonra yeni eğırlıklar hesaplanmış olur ve bu işlem init fonksiyonu içerisinde tanımlanan döngü sayısı kadar tekrar eder.
# 
# 
# 

# ##### XOR denemesi

# In[74]:


d1 = np.array([0, 1, 1, 0])


# In[83]:


perceptron.fit(X, d1)


# In[84]:


perceptron.W


# X yani input eğer yükseklik-genişlik değerleri m,n ise mxnx3 bir veri olur.40 öğrenci için imza her bir sütununda bir kişiye ait 
# imza resmi olmak üzere (m*n*3,40)'lık bir matris bizim verimiz olacaktır.Koddaki d'ye karşılık yani çıktı için 40 farklı değer olacaktır.
# ##4.Perceptron bunu tanıma yeteneğine sahip değildir.Bu yüzden bu modelin hatası elde edilemeyecektir.

# Perceptron bunu tanıma yeteneğine sahip değildir.Bu yüzden bu modelin hatası elde edilemeyecektir.


