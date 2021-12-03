# An-Attempt-at-Predicting-the-Future-with-Machine-Learning-
BTC Daily Model research Comparison of different types of LSTM architectures for Univariate Times Series(Vanilla LSTM,BILSTM ,NCP,NCPBILSTM, Encoder Decoder,Transformer)
 One mans attempt at predicting the future, one step or multistep at a time.
 
 Univariate model: 
 Drops all columns except for daily close. Pandas and numpy are used to preprocess the data,a train set and test are are created,  along with a MinMax scaller to prepare the data 
 for sequence proccessing. The data, simply a 2-dim array is proccesed by a variety of models, from the most simple Vanilla LSTM, to all the way to Transofmers, which is the last 
  exercise for this research model project.
  
  Data Source: Trading View, BINANCE_BTCUSDT Time Frame: 60 min, Range of Data: 2019 January 10th 4pm to November 28th 2021:11am
  Data Info:25207 Entries  Columns: 60 
  
  Model Breakdown:
  1. Vanilla Lstm
      ##### Basic stacked LSTM ###
basicmodel=Sequential()
basicmodel.add(LSTM(200,return_sequences=True,input_shape=(time_step,1)))
basicmodel.add(Dropout(0.001))
basicmodel.add(LSTM(200,return_sequences=True))
basicmodel.add(Dropout(0.001))
basicmodel.add(LSTM(200))
basicmodel.add(Dense(1))
basicmodel.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6))

2.####BIdirectional-LSTM ####
BImodel=Sequential()
BImodel.add(Bidirectional(LSTM(100,return_sequences=True,input_shape=(time_step,1))))
BImodel.add(Dropout(0.001))
#BImodel.add(keras.layers.RNN(ncp_cell,return_sequences=True))
BImodel.add(Bidirectional(LSTM(100,return_sequences=True)))
BImodel.add(Dropout(0.001))
BImodel.add(Bidirectional(LSTM(100)))
BImodel.add(Dense(1))
BImodel.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6))

3.###NCP LSTM##  - Neural Circuit Policy 
NCPmodel=Sequential()
NCPmodel.add(LSTM(500,return_sequences=True,input_shape=(time_step,1)))
NCPmodel.add(keras.layers.RNN(ncp_cell,return_sequences=True))
NCPmodel.add(LSTM(100,return_sequences=True))
NCPmodel.add(LSTM(100))
NCPmodel.add(Dense(1))
NCPmodel.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
3.1 Submodel : Neural Circuit Policy
ncp_wiring = kncp.wirings.NCP(
      inter_neurons=20,  # Number of inter neurons
      command_neurons=10,  # Number of command neurons
      motor_neurons=5,  # Number of motor neurons
      sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
      inter_fanout=5,  # How many outgoing synapses has each inter neuron
      recurrent_command_synapses=6,  # Now many recurrent synapses are in the    # command neuron layer
      motor_fanin=4,)  # How many incoming synapses has each motor neuron    )
      #Overwrite some of the initialization ranges
ncp_cell = LTCCell(ncp_wiring,initialization_ranges={ "w": (0.2, 2.0)},) 

4. ### BIDIRECTIONAL LSTM ATTENION NCP MODEL ###

NBmodel=Sequential()
NBmodel.add(Bidirectional(LSTM(50,return_sequences=True,input_shape=(time_step,1))))
Attention()
NBmodel.add(Bidirectional(LSTM(250,return_sequences=True)))
Attention()
NBmodel.add(keras.layers.RNN(ncp_cell,return_sequences=True))
NBmodel.add(Bidirectional(LSTM(50)))
NBmodel.add(Dense(1))
NBmodel.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))



5.### Basic AutoEncoder-Decoder  Single LSTM###
#E1D1
# n_features ==> no of features at each timestep in the data.
#
n_past=time_step
n_features=1
n_future=15
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)
#
model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)
#
model_e1d1.summary()


6. ###Encoder- Decoder  STACKED LSTM 
 ###https://colab.research.google.com/drive/1JwDl3HZ9SfvV5crtjbuFPaU_T7Q1y34w?usp=sharing
 # E2D2
# n_features ==> no of features at each timestep in the data.
#
n_past=time_step
n_features=1
n_future=15
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
#
model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e2d2.summary()  


7.  ###Encoder- Decoder  STACKED Bidirectional LSTM ###
  # E3D3
# n_features ==> no of features at each timestep in the data.
#
n_past=time_step
n_features=1
n_future=15
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.Bidirectional(LSTM(100,return_sequences = True, return_state=True))
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.Bidirectional(LSTM(100, return_state=True))
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.Bidirectional(LSTM(100, return_sequences=True))(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.Bidirectional(LSTM(100, return_sequences=True))(decoder_l1,initial_state = encoder_states2)
decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l2)
#
model_e3d3 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e3d3.summary()  


8.  ### Encoder- Decoder Bidirectional LSTM NCP ###- havent started this, 
9.  ### Transformer###- Best for last.

If you made it this far, thank you. This was a guite the journey. The univariate model is still not finished, need to make more plots and can always clean it up more. 
More importantly, I am still tinkering with a multivariate model that uses PCA along with Baysian Optimization to optimze each of the above models. 

Open note book and run the ipynb notebook, change the file name to your data and have the data in the same folder as the notebook.
