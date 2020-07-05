import numpy as np
# import random
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam

from tensorflow import keras, get_logger
from keras.layers import Input, LSTM, Dense, Masking, Embedding
from keras.models import Model, Sequential, load_model
from collections import deque
import tensorflow as tf
from matplotlib import pyplot as plt

get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)

num_units = 500
char_encoding_length = 97
batch_size = 200
epochs = 70

'''
input: the prompt to which the chatbot must respond to
output: the target response that the chatbot should learn from
'''

def train_model():

    print("Loading data file...")

    f = open("data.txt").readlines()[:2000]

    print("Building input from file...")

    printProgressBar(0, 4, prefix = 'Progress:', suffix = 'Complete', length = 50)

    pairs = np.array([(f[i], f[i + 1]) for i in range(0, len(f), 2)])
    printProgressBar(1, 4, prefix = 'Progress:', suffix = 'Complete', length = 50)

    pairs = np.random.shuffle(pairs)
    printProgressBar(2, 4, prefix = 'Progress:', suffix = 'Complete', length = 50)

    inp = [f[i] for i in range(0, len(f), 2)]
    printProgressBar(3, 4, prefix = 'Progress:', suffix = 'Complete', length = 50)

    out = [f[i + 1] for i in range(0, len(f), 2)]
    printProgressBar(4, 4, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print()
    print("One-hot encoding input...")

    progress_length = len(inp) + 2 * len(out) + 1000
    current_progress = 0

    printProgressBar(current_progress, progress_length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    encoded_input = []
    encoded_output = []
    shifted_encoded_output = []

    #fill encoded_input
    for line in inp:
        current_progress += 1
        encoded_line = []
        encoded_line.append([1 if i == char_encoding_length - 2 else 0 for i in range(char_encoding_length)]) #add start character

        for char in line:
            encoded_char = [0 for i in range(char_encoding_length)]
            encoded_char[ord(char) - 32] = 1
            encoded_line.append(encoded_char)
        encoded_input.append(encoded_line)
        printProgressBar(current_progress, progress_length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #fill encoded_output
    for line in out:
        current_progress += 1
        encoded_line = []
        encoded_line.append([1 if i == char_encoding_length - 2 else 0 for i in range(char_encoding_length)]) #add start character

        for char in line:
            encoded_char = [0 for i in range(char_encoding_length)]
            encoded_char[ord(char) - 32] = 1
            encoded_line.append(encoded_char)
            
        encoded_output.append(encoded_line)
        printProgressBar(current_progress, progress_length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #fill shifted_encoded_output        
    for line in out:
        current_progress += 1
        encoded_line = []

        for char in line:
            encoded_char = [0 for i in range(char_encoding_length)]
            encoded_char[ord(char) - 32] = 1
            encoded_line.append(encoded_char)

        encoded_line.append([1 if i == char_encoding_length - 1 else 0 for i in range(char_encoding_length)]) #add end character            
        shifted_encoded_output.append(encoded_line)
        printProgressBar(current_progress, progress_length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #pad arrays
    encoded_input = np.array(pad_array(encoded_input, char_encoding_length))
    encoded_output = np.array(pad_array(encoded_output, char_encoding_length))
    shifted_encoded_output = np.array(pad_array(shifted_encoded_output, char_encoding_length))
    printProgressBar(progress_length, progress_length, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print()
    print("Building model...")
    print()

    # Define an input sequence and process it.
    printProgressBar(0, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    encoder_inputs = Input(shape=(None, char_encoding_length))
    printProgressBar(1, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    encoder = LSTM(num_units, return_state=True)
    printProgressBar(2, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    printProgressBar(3, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    printProgressBar(4, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, char_encoding_length))
    printProgressBar(5, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(num_units, return_sequences=True, return_state=True)
    printProgressBar(6, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                        initial_state=encoder_states)
    printProgressBar(7, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    decoder_dense = Dense(char_encoding_length, activation='softmax')
    printProgressBar(8, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    decoder_outputs = decoder_dense(decoder_outputs)
    printProgressBar(9, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    printProgressBar(10, 10, prefix = 'Progress:', suffix = 'Complete', length = 50)

    print()
    print("Training model...")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
    history = model.fit([encoded_input, encoded_output], shifted_encoded_output,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)
    
    print("Saving model...")

    model.save('model')

    print("Model saved.")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def pad_array(arr, num_words):
    max_size = max([len(i) for i in arr])

    for i in range(len(arr)):
        if len(arr[i]) != max_size:
            for j in range(max_size - len(arr[i])):
                arr[i].append([0 for i in range(num_words)])

    return arr

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():
    model = load_model('model')
    encoder_inputs = model.input[0] #input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1] #input_2
    decoder_state_input_h = Input(shape=(num_units,), name='input_3')
    decoder_state_input_c = Input(shape=(num_units,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs=decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    stop = False
    predicted_message = ""
    num_decoder_tokens = decoder_inputs.shape[-1]
    print("Enter a message and the chatbot will respond:")

    while True:
        message = input(">>>")

        if message == 'quit':
            break

        encoded_message = []

        encoded_line = []
        encoded_line.append([1 if i == char_encoding_length - 2 else 0 for i in range(char_encoding_length)]) #add start character

        for char in message:
            encoded_char = [0 for i in range(char_encoding_length)]
            encoded_char[ord(char) - 32] = 1
            encoded_line.append(encoded_char)
            
        encoded_message.append(encoded_line)

        states_value = encoder_model.predict(encoded_message)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        stop = False

        while not stop:

            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

            if sampled_token_index == char_encoding_length - 1 or len(predicted_message) > 2 * len(message) or (len(predicted_message) > 7 and predicted_message[-5:] == predicted_message[-10:-5]):
                stop = True

            else:
                sampled_char = chr(sampled_token_index + 32) if sampled_token_index != char_encoding_length - 2 else ""
                predicted_message += sampled_char

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1

                # Update states
                states_value = [h, c]
            
        print(predicted_message)
        predicted_message = ""



if __name__ == '__main__':
    main()


# class DQN:
#     def __init__(self, env):
#         self.env = env
#         self.memory = deque(maxlen=2000)
        
#         self.gamma = 0.85
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
#         self.learning_rate = 0.005
#         self.tau = 0.125
#         self.num_models = 1

#         self.model = self.create_model()
#         self.target_model = self.create_model()

#     def create_model(self):
#         model = Sequential()
#         state_shape = self.env.observation_space.shape
#         model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
#         model.add(Dense(48, activation="relu"))
#         model.add(Dense(24, activation="relu"))
#         model.add(Dense(self.num_models))
#         model.compile(loss="mean_squared_error",
#             optimizer=Adam(lr=self.learning_rate))
#         return model

#     def act(self, state):
#         self.epsilon *= self.epsilon_decay
#         self.epsilon = max(self.epsilon_min, self.epsilon)
#         if np.random.random() < self.epsilon:
#             return self.env.action_space.sample()
#         return np.argmax(self.model.predict(state)[0])

#     def remember(self, state, action, reward, new_state, done):
#         self.memory.append([state, action, reward, new_state, done])

#     def replay(self):
#         batch_size = 32
#         if len(self.memory) < batch_size: 
#             return

#         samples = random.sample(self.memory, batch_size)
#         for sample in samples:
#             state, action, reward, new_state, done = sample
#             target = self.target_model.predict(state)
#             if done:
#                 target[0][action] = reward
#             else:
#                 Q_future = max(self.target_model.predict(new_state)[0])
#                 target[0][action] = reward + Q_future * self.gamma
#             self.model.fit(state, target, epochs=1, verbose=0)

#     def target_train(self):
#         weights = self.model.get_weights()
#         target_weights = self.target_model.get_weights()
#         for i in range(len(target_weights)):
#             target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
#         self.target_model.set_weights(target_weights)

#     def save_model(self, fn):
#         self.model.save(fn)

# def main():
#     env     = gym.make("MountainCar-v0")
#     gamma   = 0.9
#     epsilon = .95

#     trials  = 1000
#     trial_len = 500

#     # updateTargetNetwork = 1000
#     dqn_agent = DQN(env=env)
#     steps = []
#     for trial in range(trials):
#         cur_state = env.reset().reshape(1,2)
#         for step in range(trial_len):
#             action = dqn_agent.act(cur_state)
#             new_state, reward, done, _ = env.step(action)

#             # reward = reward if not done else -20
#             new_state = new_state.reshape(1,2)
#             dqn_agent.remember(cur_state, action, reward, new_state, done)
            
#             dqn_agent.replay()       # internally iterates default (prediction) model
#             dqn_agent.target_train() # iterates target model

#             cur_state = new_state
#             if done:
#                 break
#         if step >= 199:
#             print("Failed to complete in trial {}".format(trial))
#             if step % 10 == 0:
#                 dqn_agent.save_model("trial-{}.model".format(trial))
#         else:
#             print("Completed in {} trials".format(trial))
#             dqn_agent.save_model("success.model")
#             break