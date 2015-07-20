# -*- coding: utf-8 -*-
import os
import sys
import math
import logging
logging.basicConfig(level=logging.DEBUG)

import configure

class Feature(object):
  def __init__(self, literal, weight):
    self.literal = literal
    self.weight = weight

class FeatureSet(object):
  def __init__(self):
    self.features = {}

  def AddFeature(self, literal, weight):
    # 如果已经在集合中，就不再进行初始化
    if literal in self.features:
      return
    feature = Feature(literal, weight)
    self.features[literal] = feature

class Instance(object):
  def __init__(self, feature_list, y):
    self.feature_list = feature_list
    self.y = y

def H(feature_list):
  thelta_x = 0.0
  for feature in feature_list:
    thelta_x += feature.weight
  return 1.0 / (1.0 + math.exp(-thelta_x))

def MaximumLikelihoodEstimation(instance_list, feature_set):
  mlh = 0.0
  for instance in instance_list:
    feature_list = []
    for feature in instance.feature_list:
      if feature in feature_set.features:
        feature_list.append(feature_set.features[feature])
    h = H(feature_list)
    if h == 1.0:
      h = 0.99999999
    mlh += instance.y * math.log(h) + (1.0 - instance.y) * math.log(1.0 - h)
  return mlh

class LR(object):
  def __init__(self, category_name, train_file_path, output_model_file_path, step_len, iteration_num):
    self.category_name = category_name
    self.instance_list = []
    self.feature_set = FeatureSet()
    self.step_len = step_len
    self.iteration_num = iteration_num
    self.train_file_path = train_file_path
    self.output_model_file_path = output_model_file_path

  def ReadTrainFile(self, train_file_path):
    self.instance_list = []
    train_file = open(train_file_path)
    lines = train_file.readlines()
    train_file.close()
    logging.info("start read train file")
    read_count = 0
    for line in lines:
      tokens = line.strip().split('\t')
      try:
        y = float(tokens[-1])
        feature_list = []
        for token in tokens[:-1]:
          self.feature_set.AddFeature(token, 1.0)
          feature_list.append(token)
        self.instance_list.append(Instance(feature_list, y))
        read_count += 1
        if read_count % 1000 == 0:
          logging.info("has read " + str(read_count) + " lines")
      except:
        logging.warning("input valid: " + line.strip())
    logging.info("read train file end")

  def Iteration(self):
    for instance in self.instance_list:
      feature_list = []
      for feature in instance.feature_list:
        feature_list.append(self.feature_set.features[feature])
      delta = self.step_len * (instance.y - H(feature_list));
      for feature in instance.feature_list:
        self.feature_set.features[feature].weight += delta

  def Train(self):
    for i in range(0, self.iteration_num):
      mlh = MaximumLikelihoodEstimation(self.instance_list, self.feature_set)
      logging.info("maximum likelihood estimation in train iteration " + str(i) + " is " + str(mlh))
      self.Iteration()
    logging.info("finish trainning!")

  def OutputModel(self, output_model_file_path):
    output_model_file = open(output_model_file_path, 'w')
    logging.info("output model to " + output_model_file_path)
    for literal in self.feature_set.features:
      output_model_file.write(literal + "\t"+ str(self.feature_set.features[literal].weight) + "\n")
    output_model_file.close()

  def Run(self):
    self.ReadTrainFile(self.train_file_path)
    self.Train()
    self.OutputModel(self.output_model_file_path)

if __name__ == '__main__':
  lr = LR("美食",
          configure.train_data_file_path,
          configure.output_model_file_path,
          configure.step_len,
          configure.iteration_num)
  lr.Run()
