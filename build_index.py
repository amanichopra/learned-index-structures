from structures.b_tree import BTree, Item
import pickle
import numpy as np
import bintrees
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import types
import tempfile
import tensorflow.keras.models
import argparse
from glob import glob
from tqdm import tqdm
from timeit import default_timer as timer

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tensorflow.keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = tensorflow.keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

def get_data(path):
    with open(path, 'rb') as f:
        contents = pickle.load(f)
    return contents['data'], contents['memory']

def construct_b_tree(data, memory):
    start = timer()
    bt = BTree(2)
    for rec, memory_loc in zip(data, memory):
        bt.insert(Item(rec, memory_loc))
    elapsed_time = timer() - start
    return bt, elapsed_time

def construct_AVL(data, memory):
    start = timer()
    avl = bintrees.AVLTree()
    for rec, memory_loc in zip(data, memory):
        avl.insert(rec, memory_loc)
    elapsed_time = timer() - start
    return avl, elapsed_time

def construct_RBT(data, memory):
    start = timer()
    rbt = bintrees.RBTree()
    for rec, memory_loc in zip(data, memory):
        rbt.insert(rec, memory_loc)
    elapsed_time = timer() - start
    return rbt, elapsed_time

def construct_LR(data, memory):
    start = timer()
    reg = LinearRegression().fit(data.reshape(1, -1), memory.reshape(1, -1))
    elapsed_time = timer() - start
    return reg, elapsed_time

def construct_ANN(data, memory, epochs=10):
    start = timer()
    ann = Sequential()
    ann.add(Dense(32, input_dim=1, activation='relu'))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1))
    ann.compile(loss='mean_squared_error', optimizer='adam')
    ann.fit(data.astype(np.float32), memory.astype(np.float32), epochs=5)
    elapsed_time = timer() - start
    return ann, elapsed_time
    
def main():
    parser = argparse.ArgumentParser(description='Script to generate index models.')
    parser.add_argument('--save_path', type=str, default='./models', help='Path to save models.')
    parser.add_argument('--data_path', type=str, default='../Data', help='Path to load data from.')
    parser.add_argument('--mods', type=str, default='bt,avl,rbt,lr,ann', help='Index models to build.')
    
    args = parser.parse_args()
    
    make_keras_picklable()
    
    mods_for_data = {}
    files = glob(f'{args.data_path}/*.dat')
    for data_path in tqdm(files, total=len(files)):  
        data, memory = get_data(data_path)
        data_name = data_path.split('/')[-1].split('.')[0]
        
        print(f'Building B-Tree for {data_name}.')
        bt, bt_time = construct_b_tree(data, memory)
        print(f'Building AVL for {data_name}.')
        avl, avl_time = construct_AVL(data, memory)
        print(f'Building RB Tree for {data_name}.')
        rbt, rbt_time = construct_RBT(data, memory)
        print(f'Building Linear Regression for {data_name}.')
        lr, lr_time = construct_LR(data, memory)
        print(f'Building ANN for {data_name}.')
        ann, ann_time = construct_ANN(data, memory)
        mods = {'bt': {'mod': bt, 'train_time': bt_time}, 'avl': {'mod': avl, 'train_time': avl_time}, 'rbt': {'mod': rbt, 'train_time': rbt_time}, 'lr': {'mod': lr, 'train_time': lr_time}, 'ann': {'mod': ann, 'train_time': ann_time}}
        
        mods_for_data[data_name] = mods
    
    with open(f'{args.save_path}/models.dat', 'wb') as f:
        pickle.dump(mods_for_data, f)

if __name__ == '__main__':
    main()