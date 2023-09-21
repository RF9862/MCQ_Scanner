from cryptography.fernet import Fernet

key="K7F6dxO-u8rQitkXkp4ZaJv1sJNs_ykwt5TkkZ-pfes="

class AESCipher:
    def __init__(self):
        self.f = Fernet(key)
    
    def encrypt(self, data):
        return self.f.encrypt(bytes(data, 'utf-8'))
    
    def decrypt(self, data):
        return self.f.decrypt(data).decode("utf-8")

c = AESCipher()
data = input("Please enter Product Key : ")
print("*******  Key  ******")
print(c.encrypt(data))