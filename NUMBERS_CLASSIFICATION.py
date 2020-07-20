#!/usr/bin/env python
# coding: utf-8

# In[27]:


import tensorflow as tf
import matplotlib.pyplot as plt


# mnist = tf.keras.datasetts.mnist

# In[4]:


mnist = tf.keras.datasets.mnist


# In[58]:


from PIL import Image as Img


# In[5]:


(X_train, Y_train),(X_test, Y_test) = mnist.load_data()


# In[7]:


X_train[0]


# In[ ]:





# In[10]:


X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)


# In[34]:


plt.imshow(X_train[6], cmap= plt.cm.binary) 
plt.show()


# In[12]:


X_train[0]


# In[16]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #a simple connected layer
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))


# In[17]:


model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)


# In[20]:


val_loss,val_acc = model.evaluate(X_test, Y_test)


# In[22]:


val_loss


# In[23]:


val_acc


# In[26]:


model.save(r"C:\Users\Raza Mehndi\Desktop\digit_Classification\digit_model.model")


# In[42]:


new_model = tf.keras.models.load_model(r"C:\Users\Raza Mehndi\Desktop\digit_Classification\digit_model.model")
predictions = new_model.predict(X_test)


# In[48]:


predictions[3]


# In[54]:


plt.imshow(predictions)
plt.show()


# In[50]:


import numpy as np


# In[55]:


np.argmax(predictions[42])


# In[62]:


try:  
    img  = Image.open(r"C:\Users\Raza Mehndi\Desktop\9.jpg")  
except IOError: 
    pass
img.show()


# In[59]:


filename = input("")
with Img.open(filename) as image:
    width, height = image.size


# In[64]:


img.show()


# In[ ]:




