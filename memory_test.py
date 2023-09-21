from memory_profiler import profile
from keras.models import load_model
@profile
def my_func():
    
    model = load_model('weights/model.h5')
    del model
    print("ss")
    # return a

if __name__ == '__main__':
    for i in range(3):
        my_func()