import gym
import gym_2048
import numpy as np
import tensorflow as tf

def run_random_agent(game):
    action_space = ['up', 'down', 'left', 'right']
    i = 0
    while True:
        action = np.random.choice(action_space)
        board, reward, close, _ = game.step(action)
        print(game)
        i += 1

class SimplePolicyGradientAgent(object):
    def __init__(self):
        # Interpret board as 4x4 grid of floats.
        self.inputs = tf.placeholder('float32', (None, 4, 4))
        self.flatten_inputs = tf.layers.flatten(self.inputs)
        self.hidden1 = tf.layers.dense(
            inputs=self.flatten_inputs, units=4*4, activation=tf.nn.relu)
        self.hidden2 = tf.layers.dense(
            inputs=self.hidden1, units=8, activation=tf.nn.relu)
        self.outputs = tf.layers.dense(
            inputs=self.hidden2, units=4, activation=tf.nn.softmax)
        self.action_space = ['up', 'down', 'left', 'right']
        # Loss term.
        self.reward_inputs = tf.placeholder('float32', (None,))
        self.action_inputs = tf.placeholder('float32', (None, 4))
        self.loss = tf.losses.softmax_cross_entropy(
            self.action_inputs, self.outputs,
            reduction=tf.losses.Reduction.NONE)
        self.loss = tf.multiply(self.loss, self.reward_inputs)
        self.loss = tf.reduce_mean(self.loss)
        self.optimizer_op = tf.train.AdamOptimizer().minimize(-self.loss)
        # Config parameters (eventually move somewhere else).
        self.discount = 0.99
        self.batch_size = 16
        self.max_trajectory_steps = 1000
        # Init session.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def act(self, observation):
        # Observation is a 4x4 numpy array of floats.
        inputs = observation.reshape(1, 4, 4)
        action_probs = self.sess.run(self.outputs, {self.inputs: inputs})
        action = np.random.choice(self.action_space, p=action_probs[0])
        return action, action_probs

    # Perform one iteration of learning.
    def learn(self, env):
        # (1) Sample trajectories.
        trajectories = []
        for i in range(self.batch_size):
            print("Sampling trajectories (%d, %d)" % (i, self.batch_size))
            sample = self._sample_trajectory(env, self.max_trajectory_steps)
            trajectories.append(sample)
        # (2) Process rewards and set up terms for gradient update.
        action_sequences = map(lambda x: x[0], trajectories)
        state_sequences  = map(lambda x: x[1], trajectories)
        reward_sequences = map(lambda x: x[2], trajectories)
        cumulative_rewards = \
            map(lambda x: self._compute_discounted_rewards(x), reward_sequences)
        rewards_replicated = [[reward] * len(action_sequences[i]) \
                                 for i, reward in enumerate(cumulative_rewards)]
        actions_one_hot = \
            map(lambda x: self._action_one_hot_vectors(x), action_sequences)
        actions_one_hot = reduce(lambda x, y: x + y, actions_one_hot)
        combined_state_seqs = reduce(lambda x, y: x + y, state_sequences)
        rewards_replicated = reduce(lambda x, y: x + y, rewards_replicated)
        print("Rewards: %s" % str(cumulative_rewards))
        print("Game lengths: %s" % str([len(x) for x in action_sequences]))
        # (3) Apply gradient update.
        update_feed_dict = {
            self.inputs: combined_state_seqs,
            self.action_inputs: actions_one_hot,
            self.reward_inputs: rewards_replicated,
        }
        self.sess.run(self.optimizer_op, update_feed_dict)

    def _sample_trajectory(self, env, num_steps):
        obs = env.reset()
        actions, states, rewards = [], [], []
        for t in range(num_steps):
            action, action_probs = self.act(obs)
            obs2, reward, done, _ = env.step(action)
            # Record actions and rewards.
            actions.append(action)
            states.append(obs)
            rewards.append(reward)
            if done:
                break
            obs = obs2
        return actions, states, rewards

    def _compute_discounted_rewards(self, rewards):
        reward_sum = 0.
        for i, reward in enumerate(rewards):
            reward_sum += (self.discount**i) * reward
        return reward_sum

    def _action_one_hot_vectors(self, action_sequence):
        one_hot_mat = np.diag(np.ones(len(self.action_space)))
        one_hot_actions = []
        for action in action_sequence:
            one_hot_actions.append(one_hot_mat[self.action_space.index(action)])
        return one_hot_actions

if __name__ == '__main__':
    game = gym.make('2048-v0')
    # run_random_agent(game)
    agent = SimplePolicyGradientAgent()
    for i in range(10):
        agent.learn(game)
