# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrappers for OpenAI Gym environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import multiprocessing
import sys
import traceback

import gym
import gym.spaces
import numpy as np

import logging

logging.basicConfig(format='%(module)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

"""
在使用时，最后一层封装的动作动作空间要为无穷 [-inf, inf]
"""


class AutoReset(object):
    """Automatically reset environment when the episode is done."""

    def __init__(self, env):
        self._env = env
        self._done = True

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        if self._done:
            observ, reward, done, info = self._env.reset(), 0.0, False, {}
        else:
            observ, reward, done, info = self._env.step(action)
        self._done = done
        return observ, reward, done, info

    def reset(self):
        self._done = False
        return self._env.reset()


class ActionRepeat(object):
    """Repeat the agent action multiple steps."""

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            observ, reward, done, info = self._env.step(action)
            total_reward += reward
            current_step += 1
        return observ, total_reward, done, info


class RandomStart(object):
    """Perform random number of random actions at the start of the episode."""

    def __init__(self, env, max_steps):
        self._env = env
        self._max_steps = max_steps

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        observ = self._env.reset()
        random_steps = np.random.randint(0, self._max_steps)
        for _ in range(random_steps):
            action = self._env.action_space.sample()
            observ, unused_reward, done, unused_info = self._env.step(action)
            if done:
                logger.warning('Episode ended during random start.')
                return self.reset()
        return observ


class FrameHistory(object):
    """Augment the observation with past observations."""

    def __init__(self, env, past_indices, flatten):
        """Augment the observation with past observations.

        Implemented as a Numpy ring buffer holding the necessary past observations.

        Args:
          env: OpenAI Gym environment to wrap.
          past_indices: List of non-negative integers indicating the time offsets
            from the current time step of observations to include.
          flatten: Concatenate the past observations rather than stacking them.

        Raises:
          KeyError: The current observation is not included in the indices.
        """
        if 0 not in past_indices:
            raise KeyError('Past indices should include 0 for the current frame.')
        self._env = env
        self._past_indices = past_indices
        self._step = 0
        self._buffer = None
        self._capacity = max(past_indices)
        self._flatten = flatten

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        low = self._env.observation_space.low
        high = self._env.observation_space.high
        low = np.repeat(low[None, ...], len(self._past_indices), 0)
        high = np.repeat(high[None, ...], len(self._past_indices), 0)
        if self._flatten:
            low = np.reshape(low, (-1,) + low.shape[2:])
            high = np.reshape(high, (-1,) + high.shape[2:])
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        observ, reward, done, info = self._env.step(action)
        self._step += 1
        self._buffer[self._step % self._capacity] = observ
        observ = self._select_frames()
        return observ, reward, done, info

    def reset(self):
        observ = self._env.reset()
        self._buffer = np.repeat(observ[None, ...], self._capacity, 0)
        self._step = 0
        return self._select_frames()

    def _select_frames(self):
        indices = [(self._step - index) % self._capacity for index in self._past_indices]
        observ = self._buffer[indices]
        if self._flatten:
            observ = np.reshape(observ, (-1,) + observ.shape[2:])
        return observ


class FrameDelta(object):
    """Convert the observation to a difference from the previous observation."""

    def __init__(self, env):
        self._env = env
        self._last = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        low = self._env.observation_space.low
        high = self._env.observation_space.high
        low, high = low - high, high - low
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        observ, reward, done, info = self._env.step(action)
        delta = observ - self._last
        self._last = observ
        return delta, reward, done, info

    def reset(self):
        observ = self._env.reset()
        self._last = observ
        return observ


class RangeNormalize(object):
    """Normalize the specialized observation and action ranges to [-1, 1]."""

    def __init__(self, env, observ=None, action=None):
        self._env = env
        self._should_normalize_observ = (observ is not False and
                                         self._is_finite(self._env.observation_space))
        if observ is True and not self._should_normalize_observ:
            raise ValueError('Cannot normalize infinite observation range.')
        if observ is None and not self._should_normalize_observ:
            logger.info('Not normalizing infinite observation range.')
        self._should_normalize_action = (action is not False and
                                         self._is_finite(self._env.action_space))
        if action is True and not self._should_normalize_action:
            raise ValueError('Cannot normalize infinite action range.')
        if action is None and not self._should_normalize_action:
            logger.info('Not normalizing infinite action range.')

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = self._env.observation_space
        if not self._should_normalize_observ:
            return space
        return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

    @property
    def action_space(self):
        space = self._env.action_space
        if not self._should_normalize_action:
            return space
        return gym.spaces.Box(-np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

    def step(self, action):
        if self._should_normalize_action:
            action = self._denormalize_action(action)

        # print("action: ", action)
        # print("action range:", self._env.action_space.low, self._env.action_space.high)

        observ, reward, done, info = self._env.step(action)
        if self._should_normalize_observ:
            observ = self._normalize_observ(observ)
        return observ, reward, done, info

    def reset(self):
        observ = self._env.reset()
        if self._should_normalize_observ:
            observ = self._normalize_observ(observ)
        return observ

    def _denormalize_action(self, action):
        min_ = self._env.action_space.low
        max_ = self._env.action_space.high
        action = (action + 1) / 2 * (max_ - min_) + min_
        return action

    def _normalize_observ(self, observ):
        min_ = self._env.observation_space.low
        max_ = self._env.observation_space.high
        observ = 2 * (observ - min_) / (max_ - min_) - 1
        return observ

    def _is_finite(self, space):
        return np.isfinite(space.low).all() and np.isfinite(space.high).all()


class ClipAction(object):
    """Clip out of range actions to the action space of the environment."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = self._env.action_space.shape

        # 这里可以是真实动作空间，但需要实现动作空间归一化，所以这里暂时取的无穷范围区间
        return gym.spaces.Box(-np.inf * np.ones(shape), np.inf * np.ones(shape), dtype=np.float32)
        # 当需要采样时，使用这一句
        # return gym.spaces.Box(-np.ones(shape), np.ones(shape), dtype=np.float32)

    def step(self, action):
        action_space = self._env.action_space
        action = np.clip(action, action_space.low, action_space.high)
        return self._env.step(action)


class LimitDuration(object):
    """End episodes after specified number of steps."""

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        if self._step is None:
            raise RuntimeError('Must reset environment.')
        observ, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            self._step = None
        return observ, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ExternalProcess(object):
    """Step environment in a separate process for lock free paralellism."""

    # Message types for communication via the pipe.
    _ACTION = 1
    _RESET = 2
    _CLOSE = 3
    _ATTRIBUTE = 4
    _TRANSITION = 5
    _OBSERV = 6
    _EXCEPTION = 7
    _VALUE = 8

    def __init__(self, constructor):
        """Step environment in a separate process for lock free paralellism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.
    """
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._observ_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._observ_space:
            self._observ_space = self.__getattr__('observation_space')
        return self._observ_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
        self._conn.send((self._ATTRIBUTE, name))
        return self._receive(self._VALUE)

    def step(self, action, blocking=True):
        """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      Transition tuple when blocking, otherwise callable that returns the
      transition tuple.
    """
        self._conn.send((self._ACTION, action))
        if blocking:
            return self._receive(self._TRANSITION)
        else:
            return functools.partial(self._receive, self._TRANSITION)

    def reset(self, blocking=True):
        """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
        self._conn.send((self._RESET, None))
        if blocking:
            return self._receive(self._OBSERV)
        else:
            return functools.partial(self._receive, self._OBSERV)

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join()

    def _receive(self, expected_message):
        """Wait for a message from the worker process and return its payload.

    Args:
      expected_message: Type of the expected message.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The reveived message is not of the expected type.

    Returns:
      Payload object of the message.
    """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == expected_message:
            return payload
        raise KeyError('Received message of unexpected type {}'.format(message))

    def _worker(self, constructor, conn):
        """The process waits for actions and sends back environment results.

    Args:
      constructor: Constructor for the OpenAI Gym environment.
      conn: Connection for communication to the main process.
    """
        try:
            env = constructor()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACTION:
                    action = payload
                    conn.send((self._TRANSITION, env.step(action)))
                    continue
                if message == self._RESET:
                    assert payload is None
                    conn.send((self._OBSERV, env.reset()))
                    continue
                if message == self._ATTRIBUTE:
                    name = payload
                    conn.send((self._VALUE, getattr(env, name)))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError('Received message of unknown type {}'.format(message))
        except Exception:  # pylint: disable=broad-except
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
            logger.error('Error in environment process: {}'.format(stacktrace))
        conn.close()


class ConvertTo32Bit(object):
    """Convert data types of an OpenAI Gym environment to 32 bit."""

    def __init__(self, env):
        """Convert data types of an OpenAI Gym environment to 32 bit.

        Args:
          env: OpenAI Gym environment.
        """
        self._env = env

    def __getattr__(self, name):
        """Forward unimplemented attributes to the original environment.

        Args:
          name: Attribute that was accessed.

        Returns:
          Value behind the attribute name in the wrapped environment.
        """
        return getattr(self._env, name)

    def step(self, action):
        """Forward action to the wrapped environment.

        Args:
          action: Action to apply to the environment.

        Raises:
          ValueError: Invalid action.

        Returns:
          Converted observation, converted reward, done flag, and info object.
        """
        observ, reward, done, info = self._env.step(action)
        observ = self._convert_observ(observ)
        reward = self._convert_reward(reward)
        return observ, reward, done, info

    def reset(self):
        """Reset the environment and convert the resulting observation.

        Returns:
          Converted observation.
        """
        observ = self._env.reset()
        observ = self._convert_observ(observ)
        return observ

    def _convert_observ(self, observ):
        """Convert the observation to 32 bits.

        Args:
          observ: Numpy observation.

        Raises:
          ValueError: Observation contains infinite values.

        Returns:
          Numpy observation with 32-bit data type.
        """
        if not np.isfinite(observ).all():
            raise ValueError('Infinite observation encountered.')
        if observ.dtype == np.float64:
            return observ.astype(np.float32)
        if observ.dtype == np.int64:
            return observ.astype(np.int32)
        return observ

    def _convert_reward(self, reward):
        """Convert the reward to 32 bits.

        Args:
          reward: Numpy reward.

        Raises:
          ValueError: Rewards contain infinite values.

        Returns:
          Numpy reward with 32-bit data type.
        """
        if not np.isfinite(reward).all():
            raise ValueError('Infinite reward encountered.')
        return np.array(reward, dtype=np.float32)


class ConvertToMultiEnv(object):
    """
    将环境的动作空间和状态空间转换为多智能体环境的形式
    """

    def __init__(self, env, **kwargs):
        self._env = env
        self.args = kwargs
        self.NUM_LEGS = 4

        # 单个智能体所对应的观察空间和动作空间的大小
        # reactive 观察空间12维度，trotting 观察空间为4维, 如果加入智能体id就分别是16和8
        self.observ_len = 4
        if self.args['use_signal_in_observation']:
            self.observ_len = 12
        if self.args['use_angle_in_observation']:
            if self.args['use_signal_in_observation']:
                self.observ_len = 14
            else:
                self.observ_len = 6
        self.action_len = 2

        self._observ_space = []
        self._action_space = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        if not self._action_space:
            for i in range(self.NUM_LEGS):
                self._action_space.append(gym.spaces.Box(-np.inf * np.ones(self.action_len), np.inf * np.ones(self.action_len), dtype=np.float32))
                # self._action_space.append(self._env.action_space)

        return self._action_space

    @property
    def observation_space(self):

        if not self._observ_space:
            for i in range(self.NUM_LEGS):
                self._observ_space.append(gym.spaces.Box(-np.ones(self.observ_len), np.ones(self.observ_len), dtype=np.float32))
                # self._observ_space.append(self._env.observation_space)

        return self._observ_space

    def step(self, actions):
        multi_env_actions = np.concatenate(actions, axis=0)
        # print("actions: ", multi_env_actions)
        observ, reward, done, info = self._env.step(multi_env_actions)
        observ = self._convert_observ_reactive(observ)
        # observ = self._convert_observ_trotting(observ)
        reward = [reward for i in range(self.NUM_LEGS)]
        # done = [done for i in range(self.NUM_LEGS)]
        info = [info for i in range(self.NUM_LEGS)]
        return observ, reward, done, info

    def reset(self):
        observ = self._env.reset()
        # observ = self._convert_observ_trotting(observ)
        observ = self._convert_observ_reactive(observ)
        return observ

    def _convert_observ_reactive(self, observ):
        """

        :param observ:
        :return:
        """

        balance_observ = observ[:4]

        # 各条腿对应的电机角度，分别是左前，左后，右前和右后
        observ_left_front = balance_observ
        observ_left_back = balance_observ
        observ_right_front = balance_observ
        observ_right_back = balance_observ
        if self.args['use_signal_in_observation']:
            motor_signal = observ[:12]
            observ_left_front = motor_signal
            observ_left_back = motor_signal
            observ_right_front = motor_signal
            observ_right_back = motor_signal
        if self.args['use_angle_in_observation']:
            if self.args['use_signal_in_observation']:
                motor_angles = observ[12:].reshape(self.NUM_LEGS, 2)
            else:
                motor_angles = observ[4:].reshape(self.NUM_LEGS, 2)

            observ_left_front = np.concatenate([observ_left_front, motor_angles[0]])
            observ_left_back = np.concatenate([observ_left_back, motor_angles[1]])
            observ_right_front = np.concatenate([observ_right_front, motor_angles[2]])
            observ_right_back = np.concatenate([observ_right_back, motor_angles[3]])


        """
        # 使用原始的观察空间
        observ_left_front = observ

        observ_left_back = observ

        observ_right_front = observ

        observ_right_back = observ
        """
        """
        # 单个智能体观察的结构为[roll, pitch, roll_rate, pitch_rate
        # 将所有的电机角度作为智能体的观察
        # 当前腿的电机角度， 同对腿电机， 同侧腿电机， 对角腿电机]
        observ_left_front = np.concatenate([balance_observ, motor_angles[0],
                                            motor_angles[2], motor_angles[1], motor_angles[3]])

        observ_left_back = np.concatenate([balance_observ, motor_angles[1],
                                           motor_angles[3], motor_angles[0], motor_angles[2]])

        observ_right_front = np.concatenate([balance_observ, motor_angles[3],
                                             motor_angles[1], motor_angles[2], motor_angles[0]])

        observ_right_back = np.concatenate([balance_observ, motor_angles[2],
                                            motor_angles[0], motor_angles[3], motor_angles[1]])
        """

        """
        # 为单个智能体的观察空间添加id
        agent_id = lambda i: [0 if i != j else 1 for j in range(self.NUM_LEGS)]
        observ_left_front = np.concatenate([observ, agent_id(0)])

        observ_left_back = np.concatenate([observ, agent_id(1)])

        observ_right_front = np.concatenate([observ, agent_id(2)])

        observ_right_back = np.concatenate([observ, agent_id(3)])
        """
        """
        # 对各条腿电机的角度进行观察进行处理，使得每个智能体都有自己独特的观察空间
        observ_left_front = np.concatenate([balance_observ, np.zeros(2),
                                            motor_angles[1], motor_angles[2], motor_angles[3]])

        observ_left_back = np.concatenate([balance_observ, motor_angles[0],
                                           np.zeros(2), motor_angles[2], motor_angles[3]])

        observ_right_front = np.concatenate([balance_observ, motor_angles[0],
                                             motor_angles[1], np.zeros(2), motor_angles[3]])

        observ_right_back = np.concatenate([balance_observ, motor_angles[0],
                                            motor_angles[1], motor_angles[2], np.zeros(2)])
        """

        """
         # 只考虑当前腿和同对腿电机的角度
        observ_left_front = np.concatenate([balance_observ, motor_angles[0],motor_angles[2]])

        observ_left_back = np.concatenate([balance_observ, motor_angles[1],motor_angles[3]])

        observ_right_front = np.concatenate([balance_observ, motor_angles[3],motor_angles[1]])

        observ_right_back = np.concatenate([balance_observ, motor_angles[2],motor_angles[0]])

        """

        observs = [observ_left_front, observ_left_back, observ_right_front, observ_right_back]

        return np.array(observs)

    def _convert_observ_trotting(self, observ):

        observs = [[observ[(j + i) % self.NUM_LEGS] for j in range(self.NUM_LEGS)] for i in range(self.NUM_LEGS)]

        return np.array(observs)
