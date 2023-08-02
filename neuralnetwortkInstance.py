#This is an instance code developed by Josue Leiva to automatize celsius to fahrenheit conversions

#import libs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#array declarations to traine the machine
celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,-14,32,46,59,72,100],dtype=float)

#Use API keras to implement deep learning and make it easier instead of recreating the same API logic again and again.
capa = tf.keras.layers.Dense(units=1,input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#Execution
print("Saiyan training has about to begin...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Saiyan Training completed!")

plt.xlabel("#Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])


print("Let's make a prediction! ")
celsiusvalue = float(input("Please enter Celsius value to convert: "))
resultado = modelo.predict(([celsiusvalue]))
print("The result is: " + str(resultado)+ " fahrenheit!")
