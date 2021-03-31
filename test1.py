# from scipy.special import softmax

# import numpy as np

# x = np.array([100,70,30])

# m = softmax(x)

# print(m)


from Crypto.Cipher import AES
obj = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
message = "The answer is no"
ciphertext = obj.encrypt(message)
print(ciphertext)

obj2 = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
ciphered= obj2.decrypt(ciphertext)
print(ciphered)