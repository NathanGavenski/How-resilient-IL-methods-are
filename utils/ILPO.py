"""ILPO network for vectors."""
import shutil
from configparser import ConfigParser
import collections
from collections import deque
import glob
import json
import math
import os
import random
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from .Domains import Domain


# TODO: create better notations and rewrite the original code.


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple(
    "Model",
    "outputs, expectation, min_output, actions, gen_loss_L1, gen_grads_and_vars, train"
)


def fully_connected(inputs, n_outputs, reuse=False, scope=None):
    inputs = slim.flatten(inputs)

    return slim.fully_connected(inputs, n_outputs, activation_fn=None, reuse=reuse, scope=scope)

    # with tf.variable_scope("fc"):
    #     w_fc = weight_variable([int(inputs.shape[-1]), n_outputs])
    #     b_fc = bias_variable([n_outputs])
    #     return tf.matmul(inputs, w_fc) + b_fc


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


class VectorILPO:
    def __init__(self, config: ConfigParser, mode: str, batch_size: int, verbose: bool = False):
        self.config = config
        self.mode = mode
        self.batch_size = batch_size
        self.verbose = verbose

    def process_inputs(self, inputs):
        return inputs

    def create_ilpo(self, s_t, generator_outputs_channels):
        """Creates ILPO network."""

        # Create state embeddings.
        with tf.variable_scope("state_encoding"):
            s_t_layers = self.create_encoder(s_t)

        # Predict latent action probabilities.
        with tf.variable_scope("action"):
            flat_s = lrelu(s_t_layers[-1], .2)
            action_prediction = fully_connected(flat_s, int(self.config.get('n_actions')))

            for a in range(int(self.config.get('n_actions'))):
                tf.summary.histogram("action_{}".format(a), action_prediction[:,a])
            tf.summary.histogram("action_max", tf.nn.softmax(action_prediction))

            action_prediction = tf.nn.softmax(action_prediction)
        


        # predict next state from latent action and current state.
        outputs = []
        shape = [ind for ind in flat_s.shape]
        shape[0] = tf.shape(flat_s)[0]

        for a in range(int(self.config.get('n_actions'))):
            # there is one generator g(s,z) that takes in a state s and latent action z.
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                action = tf.one_hot([a], int(self.config.get('n_actions')))

                # obtain fully connected latent action to concatenate with state.
                action = fully_connected(action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="action_embedding")
                action = lrelu(action, .2)

                # tile latent action embedding.
                action = tf.tile(action, [1, tf.shape(s_t)[0]])
                action = tf.reshape(action, shape)

                # concatenate state and action.
                state_action = slim.flatten(tf.concat([flat_s, action], axis=-1))
                state_action = fully_connected(state_action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="state_action_embedding")
                state_action = tf.reshape(state_action, tf.shape(flat_s))

                s_t_layers[-1] = state_action
                outputs.append(self.create_generator(s_t_layers, generator_outputs_channels))

        expected_states = 0
        shape = tf.shape(outputs[0])

        # compute expected next state as sum_z p(z|s)*g(s,z)
        for a in range(int(self.config.get('n_actions'))):
            expected_states += tf.multiply(tf.stop_gradient(slim.flatten(outputs[a])), tf.expand_dims(action_prediction[:, a], -1))
        expected_states = tf.reshape(expected_states, shape)

        return (expected_states, outputs, action_prediction)

    def create_model(self, inputs, targets):
        """ Initializes ILPO model and losses."""

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        with tf.variable_scope("ilpo_loss"):
            out_channels = int(targets.get_shape()[-1])
            expected_outputs, outputs, actions = self.create_ilpo(inputs, out_channels)

            # compute loss on expected next state.
            delta = slim.flatten(targets - inputs)
            gen_loss_exp = tf.reduce_mean(
                tf.reduce_sum(tf.losses.mean_squared_error(delta, slim.flatten(expected_outputs),
                                                   reduction=tf.losses.Reduction.NONE), axis=1))

            # compute loss on min next state.
            all_loss = []

            for out in outputs:
                all_loss.append(tf.reduce_sum(
                    tf.losses.mean_squared_error(delta, slim.flatten(out),
                    reduction=tf.losses.Reduction.NONE),
                    axis=1))

            stacked_min_loss = tf.stack(all_loss, axis=-1)
            gen_loss_min = tf.reduce_mean(tf.reduce_min(stacked_min_loss, axis=1))

            gen_loss_L1 = gen_loss_exp + gen_loss_min

            # obtain images and scalars for summaries.
            tf.summary.scalar("expected_gen_loss", gen_loss_exp)
            tf.summary.scalar("min_gen_loss", gen_loss_min)

            min_index = tf.argmin(all_loss)
            min_index = tf.one_hot(min_index, int(self.config.get('n_actions')))

            shape = tf.shape(out)
            min_img = tf.stack([slim.flatten(out) for out in outputs], axis=-1)
            min_img = tf.reduce_sum(tf.multiply(min_img, tf.expand_dims(min_index, 1)), -1)
            min_img = tf.reshape(min_img, shape)


        with tf.name_scope("ilpo_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("ilpo")]
            gen_optim = tf.train.AdamOptimizer(float(self.config.get('lr')), float(self.config.get('beta1')))
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss_L1, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        return Model(
            gen_loss_L1=gen_loss_L1,
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=[inputs + out for out in outputs],
            expectation=inputs + expected_outputs,
            min_output=inputs + min_img,
            actions=actions,
            train=tf.group(gen_loss_L1, incr_global_step, gen_train),
        )

    def run(self, timesteps=None):
        """Runs training method."""

        if tf.__version__.split('.')[0] != "1":
            raise Exception("Tensorflow version 1 required")

        if not bool(self.config.get('seed')):
            seed = random.randint(0, 2**31 - 1)
        else:
            seed = int(self.config.get('seed'))

        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if not os.path.exists(self.config.get('output_dir')):
            os.makedirs(self.config.get('output_dir'))

        if not os.path.exists(self.config.get('checkpoint')):
            os.makedirs(self.config.get('checkpoint'))

        if self.mode in ["test", "export"]:
            if not bool(self.config.get('checkpoint')):
                raise Exception("checkpoint required for test mode")

            # load some options from the checkpoint
            options = {"which_direction", "ngf", "ndf", "lab_colorization"}
            with open(os.path.join(self.config.get('checkpoint'), "options.json")) as f:
                for key, val in json.loads(f.read()).items():
                    if key in options:
                        if self.verbose:
                            print("loaded", key, "=", val)
                        setattr(a, key, val)

        if self.verbose:
            for k in self.config:
                print(k, "=", self.config[k])

        examples = self.load_examples()
        self.train_examples(examples, timesteps)

    def load_examples(self):
        if self.config.get('input_dir') is None or not os.path.exists(self.config.get('input_dir')):
            raise Exception("input_dir does not exist")

        input_paths = glob.glob(os.path.join(self.config.get('input_dir'), "*.txt"))
        if len(input_paths) == 0:
            raise Exception("input_dir contains no demonstration files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the txt names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)

        with tf.name_scope("load_demonstrations"):
            paths = []
            inputs = []
            targets = []

            # Demonstrations should be of the form [s,s'] on each line in the file.
            i = 0
            for demonstration in input_paths:
                for trajectory in open(demonstration):
                    s, s_prime = trajectory.replace("\n", "").replace(",,", ",").split(" ")
                    s = eval(s)
                    s_prime = eval(s_prime)
                    inputs.append(s)
                    targets.append(s_prime)
                    paths.append(trajectory)
                    i += 1

                    if i >= 50000:
                        break

        num_samples = len(inputs)

        inputs = tf.convert_to_tensor(inputs, tf.float32)
        targets = tf.convert_to_tensor(targets, tf.float32)

        paths_batch, inputs_batch, targets_batch = tf.train.shuffle_batch(
            [paths, inputs, targets],
            batch_size=self.batch_size,
            num_threads=1,
            enqueue_many=True,
            capacity=num_samples,
            min_after_dequeue=1000)

        inputs_batch.set_shape([self.batch_size, inputs_batch.shape[-1]])
        targets_batch.set_shape([self.batch_size, targets_batch.shape[-1]])
        steps_per_epoch = int(math.ceil(num_samples / self.batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )


    def create_encoder(self, state):
        """Creates state embedding."""

        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = fully_connected(state, int(self.config.get('ngf')))
            layers.append(output)

        layer_specs = [
            int(self.config.get('ngf')) * 2,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                encoded = fully_connected(rectified, out_channels)
                layers.append(encoded)

        return layers

    def create_generator(self, layers, generator_outputs_channels):
        """Returns next state prediction given state and latent action."""

        s_t_layers = list(layers)

        with tf.variable_scope("decoder_1"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, int(self.config.get('ngf')))
            s_t_layers.append(output)

        with tf.variable_scope("decoder_2"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, generator_outputs_channels)
            s_t_layers.append(output)

        return s_t_layers[-1]

    def train_examples(self, examples, timesteps = None):
        if self.verbose:
            print("examples count = %d" % examples.count)

        # inputs and targets are [batch_size, height, width, channels]
        model = self.create_model(examples.inputs, examples.targets)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = None
        if (int(self.config.get('trace_freq')) > 0 or int(self.config.get('summary_freq')) > 0):
            logdir = self.config.get('output_dir')

        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            if self.verbose:
                print("parameter_count =", sess.run(parameter_count))

            if bool(self.config.get('checkpoint')) and os.path.exists(f'{self.config.get("checkpoint")}/checkpoint'):
                if self.verbose:
                    print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.config.get('checkpoint'))
                saver.restore(sess, checkpoint)

            max_steps = 2 ** 32
            if bool(self.config.get('max_epochs')):
                max_steps = examples.steps_per_epoch * int(self.config.get('max_epochs'))
            if bool(self.config.get('max_steps')):
                max_steps = int(self.config.get('max_steps'))
            if bool(timesteps) or timesteps == 0:
                max_steps = timesteps

            start = time.time()

            if self.verbose:
                pbar = tqdm(range(max_steps))
            else:
                pbar = range(max_steps)

            for step in pbar:
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(int(self.config.get('trace_freq'))):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(int(self.config.get('progress_freq'))):
                    fetches["gen_loss_L1"] = model.gen_loss_L1


                if should(int(self.config.get('summary_freq'))):
                    fetches["summary"] = sv.summary_op

                if should(int(self.config.get('display_freq'))):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(int(self.config.get('summary_freq'))):
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(int(self.config.get('trace_freq'))):
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(int(self.config.get('progress_freq'))):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    rate = (step + 1) * self.batch_size / (time.time() - start)

                    if self.verbose:
                        pbar.set_description_str(f'Epoch {train_epoch} image/sec {rate}')
                        pbar.set_postfix_str(f'gen_loss_L1 {results["gen_loss_L1"]}')

                if should(int(self.config.get('save_freq'))):
                    saver.save(sess, os.path.join(self.config.get('output_dir'), "model"), global_step=sv.global_step)
                    saver.save(sess, os.path.join(self.config.get('checkpoint'), "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

            saver.save(sess, os.path.join(self.config.get('output_dir'), "model"), global_step=sv.global_step)
            saver.save(sess, os.path.join(self.config.get('checkpoint'), "model"), global_step=sv.global_step)


class Policy(VectorILPO):
    def __init__(
            self,
            sess: tf.Session,
            shape: list,
            config: ConfigParser,
            batch_size: int,
            environment_name: str,
            verbose: bool = False,
            use_encoding: bool = False,
            experiment: bool = False,
            exp_writer: bool = None
    ):
        """Initializes the ILPO policy network."""

        self.sess = sess
        self.fake_sess = tf.Session()
        self.verbose = verbose
        self.use_encoding = use_encoding
        self.inputs = tf.placeholder("float", shape)
        self.targets = tf.placeholder("float", shape)
        self.state = tf.placeholder("float", shape)
        self.action = tf.placeholder("int32", [None])
        self.fake_action = tf.placeholder("int32", [None])
        self.reward = tf.placeholder("float", [None])
        self.experiment = experiment
        self.exp_writer = exp_writer
        
        self.config = config
        self.batch_size = batch_size
        self.environment_name = environment_name

        processed_inputs = self.process_inputs(self.inputs)
        processed_targets = self.process_inputs(self.targets)

        self.model = self.create_model(processed_inputs, processed_targets)

        self.action_label, loss = self.action_remap_net(self.state, self.action, self.fake_action)

        self.loss_summary = tf.summary.scalar("policy_loss", tf.squeeze(loss))
        if not experiment:
            self.reward_summary = tf.summary.scalar("reward", self.reward[0])
            self.summary_writer = tf.summary.FileWriter("policy_logs", graph=tf.get_default_graph())

        self.train_step = tf.train.AdamOptimizer(float(self.config.get('policy_lr'))).minimize(loss)

        ilpo_var_list = []
        policy_var_list = []

        # Restore ILPO params and initialize policy params.
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "ilpo" in var.name:
                ilpo_var_list.append(var)
            else:
                policy_var_list.append(var)

        saver = tf.train.Saver(var_list=ilpo_var_list)
        checkpoint = tf.train.latest_checkpoint(self.config.get('checkpoint'))
        saver.restore(sess, checkpoint)
        sess.run(tf.variables_initializer(policy_var_list))

        self.state_encoding = self.encode(processed_inputs)

    def encode(self, state):
        """Runs an encoding on a state."""

        with tf.variable_scope("ilpo_loss", reuse=True):
            with tf.variable_scope("state_encoding"):
                return self.create_encoder(state)

    def min_action(self, state, action, next_state):
        """Find the minimum action for training."""

        # Given state and action, find the closest predicted next state to the real one.
        # Use the real action as a training label for remapping the action label.
        deprocessed_outputs = [output for output in self.model.outputs]
        fake_next_states = self.sess.run(deprocessed_outputs, feed_dict={self.inputs: [state]})

        if self.use_encoding:
            next_state_encoding = sess.run(self.state_encoding, feed_dict={self.inputs: [next_state]})[0]
            fake_state_encodings = [sess.run(
                self.state_encoding,
                feed_dict={self.inputs: fake_next_state})[0] for fake_next_state in fake_next_states]
            distances = [np.linalg.norm(next_state_encoding - fake_state_encoding) for fake_state_encoding in fake_state_encodings]

        else:
            distances = [np.linalg.norm(next_state - fake_next_state) for fake_next_state in fake_next_states]
        min_action = np.argmin(distances)
        min_state = fake_next_states[min_action]

        if self.verbose:
            print("Next state", next_state)
            print("Fake states", fake_next_states)
            print("Distances", distances)
            print("Action", action)
            print("Min state", min_state)
            print("\n")

        return min_action

    def action_remap_net(self, state, action, fake_action):
        """Network for remapping incorrect action labels."""

        with tf.variable_scope("action_remap"):
            fake_state_encoding = lrelu(slim.flatten(self.create_encoder(state)[-1]), .2)
            fake_action_one_hot = tf.one_hot(fake_action, int(self.config.get('n_actions')))
            fake_action_one_hot = lrelu(fully_connected(fake_action_one_hot, int(fake_state_encoding.shape[-1])), .2)
            real_action_one_hot = tf.one_hot(action, int(self.config.get('real_actions')), dtype="int32")
            fake_state_action = tf.concat([fake_state_encoding, fake_action_one_hot], axis=-1)
            prediction = lrelu(fully_connected(fake_state_action, 64), .2)
            prediction = lrelu(fully_connected(prediction, 32), .2)
            prediction = fully_connected(prediction, int(self.config.get('real_actions')))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_action_one_hot, logits=prediction))

            return tf.nn.softmax(prediction), loss

    def P(self, state):
        """Returns the next_state probabilities for a state."""

        return self.sess.run(self.model.actions, feed_dict={self.inputs: [state]})[0]

    def greedy(self, state):
        """Returns the greedy remapped action for a state."""

        p_state = self.P(state)
        action = np.argmax(p_state)
        remapped_action = self.sess.run(self.action_label, feed_dict={self.state: [state], self.fake_action: [action]})[0]

        if self.verbose:
            print(remapped_action)

        return np.argmax(remapped_action)

    def predict(self, state):
        return self.greedy(state)

    def eval_policy(self, game, t, outside=False):
        """Evaluate the policy."""
        terminal = False
        total_reward = 0
        obs = game.reset()

        while not terminal:
            action = self.greedy(obs)
            obs, reward, terminal, _ = game.step(action)
            total_reward += reward

        if outside:
            return total_reward

        if not self.experiment:
            reward_summary = self.sess.run([self.reward_summary], feed_dict={self.reward: [total_reward]})[0]
            self.summary_writer.add_summary(reward_summary, t)
        else:
            self.exp_writer.write(str(t) + "," + str(total_reward) + "\n")

    def run_policy(self, seed: int = 0, max_steps: int = None):
        """Run the policy."""

        game = gym.make(self.environment_name)
        game.seed(seed)

        obs = game.reset()

        terminal = False
        D = deque()

        if bool(self.config.get('max_steps')):
            max_steps = int(self.config.get('max_steps'))
        if bool(max_steps):
            max_steps = max_steps

        for t in range(max_steps):

            if len(D) > 50000:
                D.popleft()

            if terminal:
                obs = game.reset()
                steps = 0.

            prev_obs = np.copy(obs)

            if np.random.uniform(0, 1) < .8:
                action = self.greedy(obs)
            else:
                action = game.action_space.sample()

            obs, reward, terminal, _ = game.step(action)
            fake_action = self.min_action(prev_obs, action, obs)

            D.append((prev_obs, action, fake_action))

            if len(D) >= self.batch_size:
                minibatch = random.sample(D, self.batch_size)
                obs_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                fake_action_batch = [d[2] for d in minibatch]

                _, loss_summary = self.sess.run([self.train_step, self.loss_summary], feed_dict={
                    self.state: obs_batch,
                    self.action: action_batch,
                    self.fake_action: fake_action_batch})

                if not self.experiment:
                    self.summary_writer.add_summary(loss_summary, t)


class ILPO:
    def __init__(self, environment_name: str, batch_size: int, verbose: bool = False):
        self.config = ConfigParser()
        self.config.read('./utils/config/ilpo.ini')
        self.env_config = self.config[Domain().given_name.upper()]
        self.environment_name = environment_name

        self.batch_size = batch_size
        self.verbose = verbose

        self.create_dataset(int(self.env_config.get('dataset_size')))

        if not os.path.exists(self.env_config.get('exp_dir')):
            os.makedirs(self.env_config.get('exp_dir'))

        if os.path.exists(self.env_config.get('checkpoint')):
            shutil.rmtree(self.env_config.get('checkpoint'))

        self.policy = None
        self.ilpo = None

    def create_dataset(self, size):
        dataset = np.load(
            f"{self.env_config.get('expert_dir')}expert_{self.environment_name}.npz",
            allow_pickle=True
        )

        beggining = np.where(dataset['episode_starts'] == True)[0]
        end = np.array([
            *np.array(beggining - 1)[1:],
            dataset['episode_starts'].shape[0]
        ])

        beggining = beggining[:size]
        end = end[:size]
        
        if not os.path.exists(f"{self.env_config.get('input_dir')}"):
            os.makedirs(f"{self.env_config.get('input_dir')}")

        with open(f"{self.env_config.get('input_dir')}/expert_{self.environment_name}.txt", "w") as f:
            self.pbar = tqdm(range(len(beggining)))
            self.pbar.set_description_str("Creating dataset")

            for b, e in zip(beggining, end):
                idx = np.array(list(range(b, e)))
                state = dataset['obs'][idx]
                nState = dataset['obs'][idx+1]
                entry = ""
                for s, nS in zip(state, nState):
                    entry += f"{str(s.tolist()).replace(' ', '')} {str(nS.tolist()).replace(' ', '')}\n"
                f.write(entry)
                    
                self.pbar.update(1)

        if self.verbose:
            self.pbar.close()

    def learn(self, total_timesteps: int = None, reset_num_timesteps: bool = False):
            tf.reset_default_graph()
            self.ilpo = VectorILPO(
                config=self.env_config,
                mode=self.config['ILPO']['mode'],
                batch_size=self.batch_size,
                verbose=self.verbose
            )
            self.ilpo.run(timesteps=total_timesteps)

            tf.reset_default_graph()
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(config=tf_config)

            self.policy = Policy(
                sess=sess,
                shape=[None, int(self.env_config.get('n_dims'))],
                config=self.env_config,
                batch_size=self.batch_size,
                environment_name=self.environment_name,
                use_encoding=False,
                verbose=self.verbose,
                experiment=True,
            )
            self.policy.run_policy(max_steps=total_timesteps)

    def predict(self, state):
        if not bool(self.policy):
            self.learn(total_timesteps=0)
        return self.policy.predict(state), None