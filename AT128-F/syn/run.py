import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected1D, Reshape, Input,Multiply,Permute,RepeatVector,Lambda,CuDNNLSTM
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling1D, MaxPooling2D,Concatenate,Add,ZeroPadding1D
from keras.layers import Embedding, LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras import regularizers, constraints
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.topology import Layer
from keras.utils import multi_gpu_model
from scipy import signal
from collections import Counter
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt


%matplotlib inline
keras.backend.clear_session()

# choose GPU card
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  
sess = tf.Session(config=config)
KTF.set_session(sess)

def get_model(trace_length, units, optimizer):

    _input = Input(shape = (trace_length,1))

    Local = LocallyConnected1D(filters=1, kernel_size=44, strides=22, padding='valid', activation=None, use_bias=True
    #                            kernel_regularizer=regularizers.l2(1e-3), 
    #                            bias_regularizer=regularizers.l2(1e-3)
                              )(_input)
    Local = BatchNormalization(axis=-2)(Local)

    Local = Reshape((-1, 2))(Local)

    FW_LSTM_out = CuDNNLSTM(units, return_sequences=True,
    #                         recurrent_regularizer=regularizers.l2(1e-5),
    #                         kernel_regularizer=regularizers.l2(1e-5),
    #                         bias_regularizer=regularizers.l2(1e-5),
                            recurrent_constraint = constraints.UnitNorm(axis=0)
                           )(Local)
    BW_LSTM_out = CuDNNLSTM(units, return_sequences=True, go_backwards = True,
    #                         recurrent_regularizer=regularizers.l2(1e-5),
    #                         kernel_regularizer=regularizers.l2(1e-5),
    #                         bias_regularizer=regularizers.l2(1e-5),
                            recurrent_constraint = constraints.UnitNorm(axis=0)
                           )(Local)
    BW_LSTM_out = Lambda(lambda xin: K.reverse(xin, axes = -2))(BW_LSTM_out)


    FW_LSTM_out_BN = BatchNormalization()(FW_LSTM_out)
    FW_LSTM_out_BN_act = Activation('tanh')(FW_LSTM_out_BN)

    FW_attention = Dense(1, use_bias=False)(FW_LSTM_out_BN_act)
    FW_attention = Flatten()(FW_attention)
    FW_attention = BatchNormalization()(FW_attention)
    FW_attention = Activation('softmax', name='FW_attention')(FW_attention)

    FW_attention = RepeatVector(units)(FW_attention)
    FW_attention = Permute([2, 1])(FW_attention)

    FW_sent_representation = Multiply()([FW_LSTM_out_BN, FW_attention])
    FW_sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(FW_sent_representation)
    FW_sent_representation =  Activation('tanh')(FW_sent_representation)

    BW_LSTM_out_BN = BatchNormalization()(BW_LSTM_out)
    BW_LSTM_out_BN_act = Activation('tanh')(BW_LSTM_out_BN)

    BW_attention = Dense(1, use_bias=False)(BW_LSTM_out_BN_act)
    BW_attention = Flatten()(BW_attention)
    BW_attention = BatchNormalization()(BW_attention)
    BW_attention = Activation('softmax', name='BW_attention')(BW_attention)

    BW_attention = RepeatVector(units)(BW_attention)
    BW_attention = Permute([2, 1])(BW_attention)

    BW_sent_representation = Multiply()([BW_LSTM_out_BN, BW_attention])
    BW_sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units,))(BW_sent_representation)
    BW_sent_representation =  Activation('tanh')(BW_sent_representation)


    FB_represent = Concatenate()([FW_sent_representation, BW_sent_representation])

    output_probabilities = Dense(256)(FB_represent)
    output_probabilities = BatchNormalization()(output_probabilities)
    output_probabilities = Activation('softmax')(output_probabilities)

    model = Model(inputs=_input, outputs=output_probabilities)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    return model

def plot_weights_of_layer1_and_att(temp_model, trs):
    
    get_att_layer_output = K.function([temp_model.layers[0].input, K.learning_phase()],
                          [temp_model.get_layer(name='FW_attention').output])
    get_att_layer_output2 = K.function([temp_model.layers[0].input, K.learning_phase()],
                          [temp_model.get_layer(name='BW_attention').output])
    
    trace_att_sum = np.zeros((temp_model.get_layer(name='FW_attention').output_shape[-1],))
    trace_att_sum2 = np.zeros((temp_model.get_layer(name='BW_attention').output_shape[-1],))
    
    for index in range(190000,190000+100):
        trs_name = '{:0>6d}.npz'.format(index)
        trs_path_name = trs + trs_name
        trace_struct = np.load(trs_path_name)
        trace = trace_struct['trace']
        trace = trace[trace_offset:trace_offset+trace_length]
        trace = np.reshape(trace,(-1, 1))
        #normalization 
        trace = trace - trace_mean
        trace = trace/trace_std
        if index==190000:
            layer_output = get_att_layer_output([[trace], 1])
            print('att_weights1_train_mode:')
            plt.figure(dpi=50)
            plt.plot(layer_output[0][0])
            plt.show() 
            layer_output2 = get_att_layer_output2([[trace], 1])
            print('att_weights2_train_mode:')
            plt.figure(dpi=50)
            plt.plot(layer_output2[0][0])
            plt.show() 
        
        # output in train mode = 1  test mode = 0
        # For BN,drop using test mod [[trace], 0]
        layer_output = get_att_layer_output([[trace], 0])
        trace_att_sum += np.asarray(layer_output[0][0])
        layer_output2 = get_att_layer_output2([[trace], 0])
        trace_att_sum2 += np.asarray(layer_output2[0][0])

    print('att_weights1_test_mode_mean:')
    plt.figure(dpi=50)
    plt.plot(trace_att_sum/100)
    plt.show() 
    print('att_weights2_test_mode_mean:')
    plt.figure(dpi=50)
    plt.plot(trace_att_sum2/100)
    plt.show()


def get_mean_std(trs_file_path, train_index, train_num, trace_offset, trace_length):
    if train_num == len(train_index):
        print('train_num check OK!')
    
    trs_file_path = trs_file_path
    trace_sum = np.zeros((trace_length,1))
    for i in train_index:
        if not i % 20000: 
            print(i)
        trs = trs_file_path + '{:0>6d}.npz'.format(i)
        trace_struc = np.load(trs)
        trace = trace_struc["trace"]
        trace = trace[trace_offset:trace_offset+trace_length]
        trace = np.reshape(trace,(-1, 1))
        trace_sum += trace
    trace_mean = trace_sum/train_num

    diff2_sum  = np.zeros((trace_length,1))
    for i in train_index:
        if not i % 20000: 
            print(i)
        trs = trs_file_path + '{:0>6d}.npz'.format(i)
        trace_struc = np.load(trs)
        trace = trace_struc["trace"]
        trace = trace[trace_offset:trace_offset+trace_length]
        trace = np.reshape(trace,(-1, 1))
        diff = trace - trace_mean
        diff2 = diff*diff
        diff2_sum += diff2
    trace_var = diff2_sum/ train_num
    trace_std = np.sqrt(trace_var)
            
    return trace_mean,trace_std

def step_decay(epoch, lr):
# #     print("step_decay_out_lr", lr)
    initial_lrate = 0.001
    drop = 0.8
    epochs_drop = 30.0
    lrate = initial_lrate *(drop ** ((1+epoch+epoch_offset)/epochs_drop))
    if lrate>0.0001:
        print("decayed_lr", lrate)
        return lrate
    else:
        print("decayed_lr", 0.0001)
        return 0.0001

def calc_GE(inter_value_pro, key_suppose):
    sorted_pro = np.sort(inter_value_pro)
    sorted_index = np.argsort(inter_value_pro)
    posi_of_key = np.where(sorted_index==key_suppose)[0]
    entropy = 256 - posi_of_key
    return entropy, sorted_pro, sorted_index  
        
def get_plaintext(test_index, trs_file_path):
    trs_file_path = trs_file_path
    length = len(test_index)
    plain_text_need = np.zeros(length)
    
    count = 0
    for i in test_index:
        trs = trs_file_path + '{:0>6d}.npz'.format(i)
        trace_struc = np.load(trs)       
        p_text = trace_struc["text"]
        sout = int(p_text[2]^p_text[7])
        sin = inv_sbox[int(sout)]
        plain_text_need[count] = int(sin ^ key_suppose)
        
        count += 1
    return plain_text_need
        
class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, index_in, batch_size, dim, n_classes, trs_file_path, 
                 trace_offset, trace_length, trace_mean, trace_std, ispredict, shuffle):
        # 'Initialization'
        self.dim = dim
        self.index_in = index_in
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.trs_file_path = trs_file_path
        self.trace_offset = trace_offset
        self.trace_length = trace_length
        self.trace_mean = trace_mean
        self.trace_std = trace_std
        self.ispredict = ispredict
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return len(self.index_in) // self.batch_size

    def __getitem__(self, index):
        # 'Generate one batch of data'
        indexes_for_batch = self.index_in[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        if self.ispredict == True:
            X = self.__data_generation(indexes_for_batch)
            return X
        else:
            X, y = self.__data_generation(indexes_for_batch)
            return X, y
        

    def on_epoch_end(self):
        # 'Updates index_in after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.index_in)

    def __data_generation(self, indexes_for_batch):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim)
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data
        count= 0
        for i in indexes_for_batch:

            trs = self.trs_file_path + '{:0>6d}.npz'.format(i)
            trace_struc = np.load(trs)
            trace = trace_struc["trace"]
            trace = trace[self.trace_offset:self.trace_offset+self.trace_length]
            trace = np.reshape(trace,(-1, 1))
            #normalization 
            trace = trace - self.trace_mean
            trace = trace/self.trace_std
            
            p_text = trace_struc["text"]
            label_value = int(p_text[2]^p_text[7])
            
            X[count,] = trace
            y[count] = label_value
            count+=1
        
        if self.ispredict == True:
            return X
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        
class SaveModelCllaBack(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if((epoch+1)%epochs_per_save==0):
            print('saving model of epoch {}'.format(epoch+1))
            temp_model = self.model
            temp_model.save('./models/Test_1_byte2_7_newest.hdf5'.format(batch_size))
            if((epoch+1)%50==0):
                temp_model.save('./models/Test_1_byte2_7_epoch{:0>6d}.hdf5'.format(epoch+1))    
        

                
            predict_sout = temp_model.predict_generator(DataGenerator(index_in=test_index, **params_predict),
                                           steps=test_num//batch_size, max_queue_size=10)
            
            predict_sout0 = predict_sout
            
            sout_array0 = predict_sout0.argmax(axis=-1)
            key_list0 = []
            
            p_array = get_plaintext(test_index, trs_file_path)
        
            for m in range(len(sout_array0)):
                sin0 = inv_sbox[sout_array0[m]]
                key_list0.append(int(sin0) ^ int(p_array[m]))
#                     if m%attack_step == 0:
            print('attack_trace:',m)
            key_counter0 = Counter(np.asarray(key_list0))
            print('key_counter_0',key_counter0.most_common(10))
            
            inter_value_pro_0 = np.zeros(256)
            pic_GE_0 = np.zeros(pic_num)
            
            for j in range(len(sout_array0)):
                
                if j>310:
                    attack_step = attack_step_large
                else :
                    attack_step = attack_step_small
                
                for key in range(256):
                    inter_value = int(Sbox[int(int(p_array[j])^int(key))])
                    inter_value_pro_0[key] += np.log(predict_sout0[j][inter_value])
                
                if j<pic_num:
                    entropy0, sorted_pro0, sorted_index0 = calc_GE(inter_value_pro_0, key_suppose)
                    pic_GE_0[j] = entropy0
                    if j%attack_step == 0:
                        print('attack_trace:',j)
                        print('entropy0:', entropy0)
                        for k in range(255,245,-1):
                            print(sorted_pro0[k], sorted_index0[k]) 
                    
                elif j%attack_step == 0:
                    entropy0, sorted_pro0, sorted_index0 = calc_GE(inter_value_pro_0, key_suppose)
                    print('attack_trace:',j)
                    print('entropy0:', entropy0)
                    for k in range(255,245,-1):
                        print(sorted_pro0[k], sorted_index0[k]) 
                        
            entropy0, sorted_pro0, sorted_index0 = calc_GE(inter_value_pro_0, key_suppose)
            print('attack_trace:',j)
            print('entropy0:', entropy0)
            for k in range(255,245,-1):
                print(sorted_pro0[k], sorted_index0[k]) 

            print('GE_curve_100_branch0:')
            plt.figure(dpi=70)
            plt.plot(pic_GE_0)
            plt.show() 
            
            plot_weights_of_layer1_and_att(temp_model,trs_file_path)

trace_offset = 0
trace_length = 46970

units = 256
my_Adam = Adam(lr=0.001)

model = get_model(trace_length, units, my_Adam)
        
batch_size = 16
epochs_per_save = 5
total_epoch = 1000

key_suppose = 153
epoch_offset = 0

train_index = np.arange(0,190000)
test_index = np.arange(190000,200000)
    
train_num = 190000
test_num = 10000
pic_num = 100

# attack_step = 100
attack_step_large = 1000
attack_step_small = 50

trs_file_path = '/home/data/disk_nvme/AT128-F/trace'

key = [107,153,173,3,79,115,220,78,245,32,187,8,33,191,168,24] 
inv_sbox = [82,9,106,213,48,54,165,56,191,64,163,158,129,243,215,251,124,227,57,130,155,47,255,135,52,142,67,68,196,222,233,203,84,123,148,50,166,194,35,61,238,76,149,11,66,250,195,78,8,46,161,102,40,217,36,178,118,91,162,73,109,139,209,37,114,248,246,100,134,104,152,22,212,164,92,204,93,101,182,146,108,112,72,80,253,237,185,218,94,21,70,87,167,141,157,132,144,216,171,0,140,188,211,10,247,228,88,5,184,179,69,6,208,44,30,143,202,63,15,2,193,175,189,3,1,19,138,107,58,145,17,65,79,103,220,234,151,242,207,206,240,180,230,115,150,172,116,34,231,173,53,133,226,249,55,232,28,117,223,110,71,241,26,113,29,41,197,137,111,183,98,14,170,24,190,27,252,86,62,75,198,210,121,32,154,219,192,254,120,205,90,244,31,221,168,51,136,7,199,49,177,18,16,89,39,128,236,95,96,81,127,169,25,181,74,13,45,229,122,159,147,201,156,239,160,224,59,77,174,42,245,176,200,235,187,60,131,83,153,97,23,43,4,126,186,119,214,38,225,105,20,99,85,33,12,125]
Sbox = [99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,4,199,35,195,24,150,5,154,7,18,128,226,235,39, 178,117,9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,83,209,0,237,32,252,177,91,106,203,190,57,74,76,88, 207,208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,81,163,64,143,146,157,56,245,188,182,218,33,16,255, 243,210,205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,231,200,55,109,141,213,78,169,108,86,244,234,101, 122,174,8,186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,112,62,181,102,72,3,246,14,97,53,87,185,134, 193,29,158,225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,140,161,137,13,191,230,66,104,65,153,45, 15,176,84,187,22]


[trace_mean, trace_std] = get_mean_std(trs_file_path, train_index, train_num, trace_offset, trace_length)


params_train = {
        'dim': (trace_length,1),
        'batch_size': batch_size,
        'n_classes': 256,
        'trs_file_path':trs_file_path,
        'trace_offset':trace_offset,
        'trace_length':trace_length,
        'trace_mean':trace_mean,
        'trace_std':trace_std,
        'ispredict':False,
        'shuffle': True}

params_valid = {
        'dim': (trace_length,1),
        'batch_size': batch_size,
        'n_classes': 256,
        'trs_file_path':trs_file_path,
        'trace_offset':trace_offset,
        'trace_length':trace_length,
        'trace_mean':trace_mean,
        'trace_std':trace_std,
        'ispredict':False,
        'shuffle': False}

params_predict = {
        'dim': (trace_length,1),
        'batch_size': batch_size,
        'n_classes': 256,
        'trs_file_path':trs_file_path,
        'trace_offset':trace_offset,
        'trace_length':trace_length,
        'trace_mean':trace_mean,
        'trace_std':trace_std,
        'ispredict':True,
        'shuffle': False}


mycallback = SaveModelCllaBack()
terminate_on_nan = keras.callbacks.TerminateOnNaN()
checkpointer_acc = ModelCheckpoint(filepath='./models/Test_1_byte2_7_best_acc.hdf5'.format(batch_size), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
checkpointer_loss = ModelCheckpoint(filepath='./models/Test_1_byte2_7_best_loss.hdf5'.format(batch_size), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
csv_logger = keras.callbacks.CSVLogger(filename='./models/tensor_log/Test_1_byte2_7_log1.csv', append=True, separator=' ')
lrate = keras.callbacks.LearningRateScheduler(step_decay)

model.fit_generator(DataGenerator(index_in=train_index, **params_train),
                    steps_per_epoch=train_num//batch_size, epochs=total_epoch, max_queue_size = 10, 
                    validation_data = DataGenerator(index_in=test_index, **params_valid),
                    validation_steps = test_num//batch_size, 
                    callbacks = [csv_logger, checkpointer_acc, checkpointer_loss, mycallback, lrate, terminate_on_nan], 
                    use_multiprocessing=True, workers = 2,verbose=1)
