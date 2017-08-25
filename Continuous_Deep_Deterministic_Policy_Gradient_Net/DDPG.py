# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING, Timothy P. Lillicrap et. al
# - memory relay
# - target_net
# - A3C
# timetag: Aug, 25, 2017
#
# Aim to solve MountainCarContinuous-v0 [https://gym.openai.com/envs/MountainCarContinuous-v0]

import tensorflow as tf
import numpy as np

class Actor(object):
    global sess
    def __init__(self, n_observation, n_action, \
                 name='actor_net', \
                 activation=tf.nn.elu, \
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                 learning_rate = 0.0001):
        self.name = name
        with tf.variable_scope(name) as scope:
            self.input = tf.placeholder(tf.float32,shape=[None,n_observation])
            self.hid1 = tf.layers.dense(self.input,200,activation=activation,kernel_initializer=kernel_initializer)
            self.hid2 = tf.layers.dense(self.hid1,400,activation=activation,kernel_initializer=kernel_initializer)
            self.output = tf.layers.dense(self.hid2,n_action,activation=tf.nn.tanh,kernel_initializer=kernel_initializer)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
            self.trainable_dict = {var.name[len(self.name):]: var \
                                   for var in trainable_vars}
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.action_grads_ph = tf.placeholder(tf.float32,shape=[None,n_action])
            theta_mu_grads = tf.gradients(self.output,trainable_vars,-self.action_grads_ph)
            self.train_op = optimizer.apply_gradients(zip(theta_mu_grads,trainable_vars))
    def predict(self,obs_batch):
        return self.output.eval(feed_dict={self.input:obs_batch}) # shape = [?,n_action]
    def train(self,obs_batch,action_grads):
        batch_size = action_grads.shape[0]
        self.train_op.run(feed_dict= \
                          {self.input:obs_batch, self.action_grads_ph:action_grads/batch_size})
    def get_trainable_dict(self):
        return self.trainable_dict
    def async_assign(self,net_2,tau=0.01):
        net_2_trainable_dict = net_2.get_trainable_dict()
        assign_op = [var.assign(var*(1-tau)+net_2_trainable_dict[key]*tau) \
                     for key, var in self.trainable_dict.items()]
        sess.run(assign_op)

class Critic(object):
    global sess
    def __init__(self, n_observation, n_action, \
                 name='critic_net', \
                 activation=tf.nn.elu, \
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), \
                 learning_rate = 0.001):
        self.name = name
        with tf.variable_scope(self.name) as scope:
            self.observation = tf.placeholder(tf.float32,shape=[None,n_observation])
            self.action = tf.placeholder(tf.float32,shape=[None,n_action])
            self.hid1 = tf.layers.dense(self.observation,128,activation=activation,kernel_initializer=kernel_initializer)
            self.hid2 = tf.concat([self.hid1,self.action],axis=1)
            self.hid3 = tf.layers.dense(self.hid2,256,activation=activation,kernel_initializer=kernel_initializer)
            self.Q = tf.layers.dense(self.hid3,1,kernel_initializer=kernel_initializer)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
            self.trainable_dict = {var.name[len(self.name):]: var \
                                       for var in trainable_vars}
            self.action_grads = tf.gradients(self.Q, self.action)
            self.Qexpected_ph = tf.placeholder(tf.float32,shape=[None,1])
            self.loss = tf.losses.mean_squared_error(self.Q,self.Qexpected_ph)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)
    def predict(self,obs_batch, action_batch):
        return self.Q.eval(feed_dict={self.observation: obs_batch, self.action:action_batch})
    def train(self,obs_batch,action_batch,Qexpected_batch):
        self.train_op.run(feed_dict= \
                          {self.observation:obs_batch,self.action:action_batch,self.Qexpected_ph:Qexpected_batch})
    def get_trainable_dict(self):
        return self.trainable_dict
    def async_assign(self,net_2,tau=0.01):
        net_2_trainable_dict = net_2.get_trainable_dict()
        assign_op = [var.assign(var*(1-tau)+net_2_trainable_dict[key]*tau) \
                     for key, var in self.trainable_dict.items()]
        sess.run(assign_op)
    def get_action_grads(self,obs_batch,action_batch):
        return sess.run(self.action_grads, feed_dict={self.observation:obs_batch, self.action:action_batch})[0]

class ActorCritic(object):
    def __init__(self, n_observation, n_action, \
                 name='',gamma=0.99):
        self.name = name
        self.gamma = 0.99
        self.actor = Actor(n_observation, n_action,name='actor{}'.format(self.name))
        self.critic = Critic(n_observation, n_action, name='critic{}'.format(self.name))
    def predict(self,obs_batch): # from oberservation to Q prediction
        action_val = self.actor.predict(obs_batch)
        return self.critic.predict(obs_batch,action_val)
    def train(self,target_net,memory_batch):
        extract_mem = lambda k : np.array([item[k] for item in memory_batch])
        obs_batch = extract_mem(0)
        action_batch = extract_mem(1)
        reward_batch = extract_mem(2)
        next_obs_batch = extract_mem(3)
        done_batch = extract_mem(4)
        Qnext_batch = target_net.predict(next_obs_batch)
        Qexpected_batch = reward_batch+self.gamma*(1-done_batch)*Qnext_batch.ravel()
        Qexpected_batch = np.expand_dims(Qexpected_batch,1)
        self.critic.train(obs_batch,action_batch,Qexpected_batch)
        action_grads = self.critic.get_action_grads(obs_batch,action_batch)
        self.actor.train(obs_batch,action_grads)
        target_net.async_assign(self)
    def async_assign(self,net_2):
        self.actor.async_assign(net_2.actor)
        self.critic.async_assign(net_2.critic)

from collections import deque
class Memory(object):
    def __init__(self,memory_size=10000,batch_size=256):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
    def __len__(self):
        return len(self.memory)
    def append(self,item):
        self.memory.append(item)
    def sample_batch(self,batch_size=0):
        batch_size = batch_size if batch_size>0 else self.batch_size
        assert(batch_size<=len(self.memory))
        idx = np.random.permutation(len(self.memory))
        return [self.memory[i] for i in idx]

if __name__ == '__main__':
	import gym
	max_episode = 300
	memory_warmup = 1000
	save_path = 'DDPG_net_Class.ckpt'
	tf.reset_default_graph()
	ac = ActorCritic(2,1)
	ac_target = ActorCritic(2,1,name='_target')
	iteration = 0
	episode = 0
	episode_steps = 0
	env = gym.make('MountainCarContinuous-v0')
	obs = env.reset()
	memory = Memory()
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    init.run()
	    while episode < max_episode:
	        print('\riter {}, ep {}, ep steps {}'.format(iteration,episode,episode_steps),end='')
	        action = ac.actor.predict(np.reshape(obs,[1,-1]))[0]
	        next_obs, reward, done,info = env.step(action)
	        memory.append([obs,action,reward,next_obs,done])
	        if iteration >= memory_warmup:
	            memory_batch = memory.sample_batch()
	            ac.train(ac_target,memory_batch)
	            ac_target.async_assign(ac)
	        iteration += 1
	        episode_steps += 1
	        if done:
	            obs_batch = np.array([item[0] for item in memory_batch])
	            print(', Q_average {}'.format(np.mean(ac.predict(obs_batch))))
	            obs = env.reset()
	            episode += 1
	            episode_steps = 0
	            if episode%5 == 0:
	                saver.save(sess,save_path)
	        else:
	            obs = next_obs