import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import gym

# Get shape
GYM_NAME = 'CartPole-v0'
env = gym.make(GYM_NAME)
obs_shape = env.observation_space.shape
n_action = env.action_space.n
env.close()

def create_q_net(X, name=None):
    with tf.variable_scope(name) as scope:
        he_init = tf.contrib.layers.variance_scaling_initializer()
        xavier_init=tf.contrib.layers.xavier_initializer()
#         X = tf.placeholder(tf.float32, shape=(None,)+obs_shape)
        dense1 = tf.layers.dense(X,20,kernel_initializer=he_init, activation=tf.nn.elu)
#         dense1 = tf.layers.dense(X,20,activation=tf.nn.elu)
        dense2 = tf.layers.dense(dense1,10,kernel_initializer=he_init, activation=tf.nn.elu) # not used
        q_net = tf.layers.dense(dense1, n_action, kernel_initializer=xavier_init)
        trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
    return q_net, trainable_vars

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None,)+obs_shape)
q_net, _ = create_q_net(X, name='q_network')

from collections import deque
learning_rate = 0.001
gamma = 0.999
memory_cap = 1000
max_iteration= 100000
batch_size = 100
n_step = 1
memory_warmup_size = memory_cap

action_ph = tf.placeholder(tf.int32, shape=[None,])
qn_ph = tf.placeholder(tf.float32, shape=[None,])
q0 = tf.reduce_sum(q_net*tf.one_hot(action_ph,n_action),axis=1)
loss = tf.reshape(tf.squared_difference(q0,qn_ph),[-1,1]) #CHANGED
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

def epsilon_greedy(q_net_val, step):
    eps_min = 0.05
    eps_max = 1.0
    eps_decay_steps = 50000
#     epsilon = eps_min + step/eps_decay_steps*(eps_max-eps_min)
    epsilon =  max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_action)
    return np.argmax(q_net_val)
    
def check_model():
    import time
    env = gym.make('CartPole-v0')
    obs = env.reset()
    step = 0
    while True:
        q_net_val = q_net.eval(feed_dict={X: np.reshape(obs,[-1,4])})
        curr_action = epsilon_greedy(q_net_val,max_iteration)
        env.render()
#         time.sleep(0.1)
        obs, _,done,_ = env.step(curr_action)
        step += 1
        if done:
            break
    env.close()
    return step

init = tf.global_variables_initializer()
env = gym.make(GYM_NAME)
from gym import wrappers
env = wrappers.Monitor(env,'./tmp/',force=True)
prev_obs = env.reset()
prev_action = env.action_space.sample()
memory = deque(maxlen=memory_cap)
iteration = 0
episode = 0
train_step = 0
config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
with tf.Session(config=config) as sess:
    init.run()
    while train_step < 100000:
        print('\riteration {}, episode = {}, train_step {}'.format(iteration, episode, train_step),end='')
        obs, reward, done, _ = env.step(prev_action)
        q_net_val = q_net.eval(feed_dict = {X: np.expand_dims(obs,0)})
        memory.append([prev_obs, prev_action, reward, np.max(q_net_val),done])
        prev_obs, prev_action = obs, epsilon_greedy(q_net_val, train_step) # CHANGED
        if iteration > memory_warmup_size: # train
            idx = np.random.permutation(len(memory)-1)[:batch_size]
            X_batch = np.array([memory[b][0] for b in idx])
            action_batch = np.array([memory[b][1] for b in idx])
            reward_batch = np.array([memory[b][2] for b in idx])
            q_batch = np.array([memory[b][3] for b in idx])
            done_batch = np.array([memory[b][4] for b in idx])
            qn_batch = reward_batch+(~done_batch)*q_batch*gamma
            train_op.run(feed_dict = {X:X_batch, action_ph:action_batch, qn_ph:qn_batch})
            train_step += 1
        if done:
            prev_obs = env.reset()
            episode += 1
            if episode%100==0:
                check_model()
        iteration += 1
env.close()