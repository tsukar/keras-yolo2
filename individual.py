import random
import itertools
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Dropout, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Activation, BatchNormalization
from keras.layers import Add, Concatenate
from keras.layers import Dense

class SerializedLayer:
  def __init__(self, func, bypass_index=None):
    self.class_name = func.__class__.__name__
    self.config = func.get_config()
    self.bypass_index = bypass_index
    self.output_shape = None

  def evaluate(self, prev_output):
    func = self.deserialize()
    if self.class_name != 'Conv2D':
      return func(prev_output)
    else:
      return Activation('relu')(BatchNormalization()(func(prev_output)))

  def deserialize(self):
    return layers.deserialize({'class_name': self.class_name,
                               'config': self.config})

class Individual(list):
  def __init__(self):
    self.last_applied_operation = 'initialize'

  def sample_layer(self, class_name):
    candidates = [l for l in self if l.class_name == class_name]
    if candidates:
      chosen = random.choice(candidates)
      return chosen
    else:
      raise RuntimeError('No {} layer is found.'.format(class_name))

  def remove_layer(self, class_name):
    chosen = self.sample_layer(class_name)
    chosen_index = self.index(chosen)

    offset = 1
    for i in range(chosen_index + 1, len(self)):
      if self[i].bypass_index == chosen_index:
        self.pop(i)
        offset += 1
      elif self[i].bypass_index != None:
        if self[i].bypass_index > chosen_index:
          self[i].bypass_index -= 1
    self.pop(chosen_index)

  def get_possible_pairs(self, ignore_channel=False):
    possible_pairs = []
    print(self)
    for pair in itertools.combinations(self, 2):
      l1, l2 = pair

      print('l1:', l1.output_shape[1:])
      print('l2:', l2.output_shape[1:])
      if l1.output_shape == None or l2.output_shape == None:
        raise RuntimeError('Output shape has not yet been defined.')

      print(l1.output_shape[1:] == l2.output_shape[1:])
      if (l1.output_shape[1:] == l2.output_shape[1:]) or \
         (ignore_channel and (l1.output_shape[1:-1] == l2.output_shape[1:-1])):
        possible_pairs.append(sorted([self.index(l1), self.index(l2)]))

    if possible_pairs:
      return possible_pairs
    else:
      raise RuntimeError('No possible pair is found.')

def add_convolution(ind):
  position = random.randrange(len(ind) + 1)
  conv_layer = SerializedLayer(Conv2D(1024, kernel_size=(3, 3),
                               strides=1,
                               padding='same',
                               use_bias=False))
  ind.insert(position, conv_layer)

def remove_convolution(ind):
  ind.remove_layer('Conv2D')

def alter_channel_number(ind):
  conv_layer = ind.sample_layer('Conv2D')
  conv_layer.config['filters'] = random.choice([512, 1024, 2048])

def alter_filter_size(ind):
  conv_layer = ind.sample_layer('Conv2D')
  conv_layer.config['kernel_size'] = random.choice([(1, 1), (3, 3), (5, 5)])

def alter_stride(ind):
  conv_layer = ind.sample_layer('Conv2D')
  conv_layer.config['strides'] = random.choice([(1, 1), (2, 2)])

def add_dropout(ind):
  dropout = SerializedLayer(Dropout(0.5))
  fc_layer = ind.sample_layer('Dense')
  position = ind.index(fc_layer)
  ind.insert(position, dropout)

def remove_dropout(ind):
  ind.remove_layer('Dropout')

def add_pooling(ind):
  conv_layer = ind.sample_layer('Conv2D')
  position = ind.index(conv_layer) + 1
  pool_layer = SerializedLayer(MaxPooling2D(pool_size=(2, 2),
                               strides=2,
                               padding='same'))
  ind.insert(position, pool_layer)

def remove_pooling(ind):
  ind.remove_layer('MaxPooling2D')

def add_skip(ind):
  pair = random.choice(ind.get_possible_pairs())
  print(1, pair[1])
  print(0, pair[0])
  ind.insert(pair[1], SerializedLayer(Add(), bypass_index=pair[0]))

def remove_skip(ind):
  ind.remove_layer('Add')

def add_concatenate(ind):
  pair = random.choice(ind.get_possible_pairs(ignore_channel=True))
  print(1, pair[1])
  print(0, pair[0])
  ind.insert(pair[1], SerializedLayer(Concatenate(), bypass_index=pair[0]))

def remove_concatenate(ind):
  ind.remove_layer('Concatenate')

def add_fully_connected(ind):
  dim = random.choice([50, 100, 150, 200])
  fc_layer = SerializedLayer(Dense(dim))

  possible_positions = \
    [i + 1 for i, l in enumerate(ind) if l.class_name == 'Dense']
  last_position = len(ind)
  if last_position not in possible_positions:
    possible_positions.append(last_position)

  position = random.choice(possible_positions)
  ind.insert(position, fc_layer)

def remove_fully_connected(ind):
  ind.remove_layer('Dense')
