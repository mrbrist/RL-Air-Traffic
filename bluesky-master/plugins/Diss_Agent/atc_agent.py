"""
    Ellis Thompson - Undergraduate Dissertation BSc Swansea University
    Based on the system by Marc Brittian (https://github.com/marcbrittain/bluesky/tree/master)
    May 2020
"""

import geopy
import keras
#import numba as nb
import numpy as np
import tensorflow as tf
from bluesky.tools import geo
from bluesky.tools.aero import ft
import keras.backend as K
import datetime

GAMMA = 0.9
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 32
CLIPPING = 0.2
LOSS = 0.00001

#@nb.njit()
def discount(r, discounted_r, cum_r):
    for t in range(len(r) - 1, -1, -1):
        cum_r = r[t] + cum_r * GAMMA
        discounted_r[t] = cum_r
    return discounted_r

def dist_goal(traf,_id):
    _id = traf.id2idx(_id)

    olat = traf.lat[_id]
    olon = traf.lon[_id]
    ilat,ilon = traf.ap.route[_id].wplat[0],traf.ap.route[_id].wplon[0]

    dist = geo.latlondist(olat,olon,ilat,ilon)/geo.nm
    return dist

# Get the nearest aircraft distance
def nearest_ac(self, dist_matrix, _id, traf):
    row = dist_matrix[:,_id]
    close = 10e+25
    alt_separations = 0

    for i, dist in enumerate(row):
        if i == _id:
            continue

        if dist < close:
            close = dist
            this_alt = traf.alt[_id]
            close_alt = traf.alt[i]
            alt_separations = abs(this_alt - close_alt)
            

    return close, alt_separations

# Get the distance matrix for all aircraft
def get_distance_matrix_ac(self, traf):
    size = traf.lat.shape[0]
    distances = geo.latlondist_matrix(np.repeat(traf.lat, size), np.repeat(traf.lon, size), np.tile(traf.lat, size), np.tile(traf.lon, size)).reshape(size, size)
    
    return distances

# PPO loss function
def PPO_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - CLIPPING, max_value=1 + CLIPPING) * advantage) + LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

class Mult_Agent:
    def __init__(self, state_size, no_actions, action_size, num_intruders, routes):
        self.experience = {}
        self.dist_close = {}
        self.dist_goal = {}
        self.statesize = state_size
        self.no_actions = no_actions
        self.action_size = action_size
        self.alts = [24000, 26000, 28000, 30000, 32000]
        self.num_intruders = num_intruders
        self.value_size = 1
        self.routes = routes
        self.routeDistances()
        self.max_speed = 253
        self.min_speed = 118
        self.max_alt = 9143
        self.min_alt = 7925
        self.no_routes = len(routes)
        self.max_agents = 0

        self.model = self.__Build__Model()
        
        if not (self.load('best_model_Sim 2-3.h5')):
            #self.load('latest_model_model2.h5')
            pass

    # Build the ml model
    def __Build__Model(self):
        # Input Layers
        _input = tf.keras.layers.Input(shape=(self.statesize,), name='input_states')

        _input_context = tf.keras.layers.Input(shape=(self.num_intruders, 5), name='context')
        
        empty = tf.keras.layers.Input(shape=(HIDDEN_SIZE,), name='empty')
        
        advantage = tf.keras.layers.Input(shape=(1,),name='A')
        
        old_prediction = tf.keras.layers.Input(shape=(self.action_size,), name='prev_predictions')

        flatten_context = tf.keras.layers.Flatten()(_input_context)

        # Hidden Layers
        Hidden_1 = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(flatten_context)
        # Combine the encoded H1 and input
        combined = tf.keras.layers.concatenate([_input, Hidden_1], axis=1)

        Hidden_2 = tf.keras.layers.Dense(256, activation='relu')(combined)

        Hidden_3 = tf.keras.layers.Dense(256, activation='relu')(Hidden_2)

        # Output Layer
        output = tf.keras.layers.Dense(self.action_size + 1, activation=None)(Hidden_3)
        
        # Split output into policy anmd value outputs
            ## Policy out
        policy = tf.keras.layers.Lambda(lambda x: x[:,:self.action_size], output_shape=(self.action_size,))(output)
            ## Value out
        value = tf.keras.layers.Lambda(lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)
        
        # Apply activation function to the output layers
        policy_out = tf.keras.layers.Activation('softmax', name='policy_out')(policy)
        value_out = tf.keras.layers.Activation('linear', name='value_out')(value)

        optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

        model = tf.keras.models.Model(inputs=[_input, _input_context, empty, advantage, old_prediction], outputs=[policy_out, value_out])
        
        self.estimator = tf.keras.models.Model(inputs=[_input, _input_context, empty], outputs=[policy_out, value_out])
        

        model.compile(optimizer=optimizer, loss={'policy_out': PPO_loss(advantage=advantage, old_prediction=old_prediction),
        'value_out': 'mse'})
        
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime(" %Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        print(model.summary())
        return model

    # Load the weights
    def load(self, file):
        print('Attempting to load Weights from {}...'.format(file))
        try:
            self.model.load_weights(file)
            print('Weights loaded successfully from {}'.format(file))
            return True
        except Exception as ex:
            print(ex)
            print('Error loading weights from {}'.format(file))
            return False

    # Save the weights
    def save(self, highest=False, _type='default'):
        if highest:
            self.model.save_weights('best_model_{}.h5'.format(_type))
        else:
            self.model.save_weights('latest_model_{}.h5'.format(_type))

    # perform an action
    def act(self, state, context):
        context = context.reshape((state.shape[0], -1, 5))
        
        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:,:]
        if context.shape[1] < self.num_intruders:
            context = tf.keras.preprocessing.sequence.pad_sequences(context, self.num_intruders, dtype='float32')
        
        policy, value = self.estimator.predict({'input_states': state, 'context': context, 'empty': np.zeros((state.shape[0], HIDDEN_SIZE))}, batch_size=state.shape[0])
        return policy, value


    def normalize_alt(self, alt):
        if alt > self.max_alt:
            self.max_alt = alt
        
        if alt < self.min_alt:
            self.min_alt = alt
        
        return(alt-self.max_alt)/(self.max_alt-self.min_alt)

    def normalize_state(self, state, id_):
        goal_d = self.dist_goal[id_] / self.max_d
        alt = self.normalize_alt(state[2])
        route = state[3] / (self.no_routes - 1)
        vs = state[4]
        
        norm_array = np.array([goal_d, alt, route, vs, 3 / self.max_d])
        return norm_array
    
    def normalize_context(self, cur_state, context, state, id_):
        own_route = int(state[3])

        goal_distance = self.dist_goal[id_] / self.max_d
        alt = self.normalize_alt(state[2])
        route = state[3] / (self.no_routes - 1)
        vs = state[4]
        route_int = int(context[3])

        if own_route == route_int:
            dist_away = abs(state[0] - goal_distance)
        else:
            d  = geo.latlondist(state[0],state[1],context[0],context[1])/geo.nm
            dist_away = d / self.max_d
        
        context_array = np.array([goal_distance, alt, route, vs, dist_away])

        return context_array.reshape(1,1,5)

    # Get a terminal update for a given aircraft
    def update(self, traf, _id, routes):
        # If the ac is terminal
        T = 0
        # The type of terminal that the ac is
        # 0 = not
        # 1 = collision
        # 2 = goal reached
        type_ = 0

        distance, v_separation = nearest_ac(self, get_distance_matrix_ac(self, traf), _id, traf)

        # Using 2000ft for verticle separation 2000ft ~ 609m
        if distance <= 3 and v_separation / ft < 1000:
            T = True
            type_ = 1
        
        self.dist_close[traf.id[_id]] = distance

        goal_d = dist_goal(traf, traf.id[_id])

        if goal_d < 5 and T == 0:
            T = True
            type_ = 2
        
        self.dist_goal[traf.id[_id]] = goal_d

        return T, type_
    
    def store(self, state, action, next_state, traf, id_, ac_routes, terminal_type = 0):
        reward = 0
        done = False

        index = traf.id2idx(id_)

        dist, alt = nearest_ac(self, get_distance_matrix_ac(self, traf), index, traf)

        alt = alt / ft
        #print(f'{id_}, {terminal_type}, {alt}, {dist}')
        #print(terminal_type)
        if terminal_type == 0:
            if (dist > 5 and alt < 995) and (len(traf.id) > 1):
                reward = 0 - 0.05 * (alt / 100)
            elif dist > 5 and alt > 995:
                reward = 0 + 0.05 * (alt / 100)
            
            if dist <= 3 and alt >= 1000:
                reward = 0.5

        if terminal_type == 1:
            reward = -1
            done = True
        
        if terminal_type == 2:
            reward = 1
            done = True

        state, context = state
        state = state.reshape((1, 5))
        context = context.reshape((1, -1, 5))

        if context.shape[1] > self.num_intruders:
            context = context[:, -self.num_intruders:,:]
            
        self.max_agents = max(self.max_agents, context.shape[1])

        if not id_ in self.experience.keys():
            self.experience[id_] = {}
        
        try:
            self.experience[id_]['state'] = np.append(self.experience[id_]['state'], state, axis=0)
            
            if self.max_agents > self.experience[id_]['context'].shape[1]:
                self.experience[id_]['context'] = np.append(tf.keras.preprocessing.sequence.pad_sequences(self.experience[id_]['context'], self.max_agents, dtype='float32'), context, axis=0)
            else:
                self.experience[id_]['context'] = np.append(self.experience[id_]['context'], tf.keras.preprocessing.sequence.pad_sequences(context, self.max_agents, dtype='float32'), axis=0)
            
            self.experience[id_]['action'] = np.append(self.experience[id_]['action'],action)
            self.experience[id_]['reward'] = np.append(self.experience[id_]['reward'],reward)
            self.experience[id_]['done'] = np.append(self.experience[id_]['done'], done)
        except:
            self.experience[id_]['state'] = state
            if self.max_agents > context.shape[1]:
                self.experience[id_]['context'] = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
            else:
                self.experience[id_]['context'] = context

            self.experience[id_]['action'] = [action]
            self.experience[id_]['reward'] = [reward]
            self.experience[id_]['done'] = [done]


    def routeDistances(self):
        self.route_distances = []

        for pos in self.routes:
            olat, olon, _, glat, glon = pos
            _, distance = geo.qdrdist(olat, olon, glat, glon)
            self.route_distances.append(distance)
        
        self.max_d = max(self.route_distances)
    
    def train(self):

        """Grab samples from batch to train the network"""

        total_state = []
        total_reward = []
        total_A = []
        total_advantage = []
        total_context = []
        total_policy = []

        total_length = 0

        for transitions in self.experience.values():
            episode_length = transitions['state'].shape[0]
            total_length += episode_length

            state = transitions['state']#.reshape((episode_length,self.state_size))
            context = transitions['context']
            reward = transitions['reward']
            done = transitions['done']
            action  = transitions['action']

            discounted_r, cumul_r = np.zeros_like(reward), 0
            discounted_rewards = discount(reward,discounted_r, cumul_r)
            policy,values = self.estimator.predict({'input_states':state,'context':context,'empty':np.zeros((len(state),HIDDEN_SIZE))},batch_size=256)
            advantages = np.zeros((episode_length, self.action_size))
            index = np.arange(episode_length)
            advantages[index,action] = 1
            A = discounted_rewards - values[:,0]

            if len(total_state) == 0:

                total_state = state
                if context.shape[1] == self.max_agents:
                    total_context = context
                else:
                    total_context = tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32')
                total_reward = discounted_rewards
                total_A = A
                total_advantage = advantages
                total_policy = policy

            else:
                total_state = np.append(total_state,state,axis=0)
                if context.shape[1] == self.max_agents:
                    total_context = np.append(total_context,context,axis=0)
                else:
                    total_context = np.append(total_context,tf.keras.preprocessing.sequence.pad_sequences(context,self.max_agents,dtype='float32'),axis=0)
                total_reward = np.append(total_reward,discounted_rewards,axis=0)
                total_A = np.append(total_A,A,axis=0)
                total_advantage = np.append(total_advantage,advantages,axis=0)
                total_policy = np.append(total_policy,policy,axis=0)


        total_A = (total_A - total_A.mean())/(total_A.std() + 1e-8)
        self.model.fit({'input_states':total_state,'context':total_context,'empty':np.zeros((total_length,HIDDEN_SIZE)),'A':total_A,'prev_predictions':total_policy}, {'policy_out':total_advantage,'value_out':total_reward}, shuffle=True,batch_size=total_state.shape[0],epochs=8, verbose=0, callbacks=[self.tensorboard_callback])


        self.max_agents = 0
        self.experience = {}
            


        
