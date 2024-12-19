import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from collections import deque

class Agent:

	rewards = []
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.delta = 0.97
		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

		return model

	def act(self, state):
        #Sample random action in the first episodes
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])


	def stockRewards(self, rewardto) :
		self.rewards.append(rewardto)


	def expReplay(self, batch_size):

		mini_batch = []
		l = len(self.memory)
		advantage_value = 0
		index = l - batch_size + 1


		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay


	def getRewards(self):

		mini_batch = []
		rewards = []
		l = len(self.memory)

		for i in range(0, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			if reward > 0 :
				rewards.append(reward)

		return rewards


	def getAgentsrewards(self):
		return 	self.rewards



import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
	vec = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()

	for line in lines[1:]:
		vec.append(float(line.split(",")[4]))

	return vec

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


# returns an n-day state representation ending at time t
def getState(data, t, n):
	d = t - n + 1
	block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])


def loadData(stockname):
    data = getStockDataVec(stockname)
    print(len(data))
    state = getState(data, 0, 4)
    t=0
    d = t - 4

    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    print('------------ Minus')
    print(-d * [data[0]] + data[0:t + 1]    )
    print('------------ State')
    print(state)
    print('------------  Block')
    res = []
    for i in range(3):
        res.append(sigmoid(block[i + 1] - block[i]))
    print(block)
    return 0

#loadData("GOLD")

import sys

total_profitl = []
buy_info = []
sell_info = []
data_Store = []

stock_name, window_size, episode_count = 'GOLD', 3, 10

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):

	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
        #Sample a Random action in the first episodes
        #and then try to predict the best action for a given state
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print("Buy: " + formatPrice(data[t]))

            #save results for visualisation
			buy_info.append(data[t])
			d = str(data[t]) + ', ' + 'Buy'
			data_Store.append(d)

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price

			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			total_profitl.append(data[t] - bought_price)

			step_price = data[t] - bought_price

			info = str(data[t]) +',' + str( step_price) +',' + str(reward)
			sell_info.append(info)
			d = str(data[t]) + ', ' + 'Sell'
			data_Store.append(d)



		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))