#!/usr/bin/env python
# coding=utf-8

# Author      :   Xionghui Chen
# Created     :   2017.6.29
# Modified    :   2017.7.23
# Version     :   1.0

import os
import logging
import csv
import random
import numpy as np 
import copy
import tensorflow as tf 
import tensorflow.contrib.layers as layers

from Base_Data_Structure import DataFeature
# from Base_Agent import DqnAgent

import combine_baselines.common.tf_util as U
from combine_baselines import logger
from combine_baselines import deepq
from combine_baselines.deepq.models import model,mlp
from combine_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from combine_baselines.common.schedules import LinearSchedule
# dtree
from Global_Variables import  ATTR_TYPE_CONTINUOUS, MAX_VALUE
# network
from Global_Variables import REPLAY_MEMORY, OBSERVE,MODEL_NAME, CPU_NUM, SAVE_MODEL_INTERVAL, UPDATE_TARGET_INTERVAL, SCHEDULE_TIMES
from Global_Function import dic_to_list,list_to_dic
from Global_Variables import ACTION, FINAL_EPSILON, PRECISION
from Global_Variables import MODEL_PREFIX_PATH

class ForestAgent(object):
    def __init__(self, data, config, exp_file_name, itera_times, model_type=[372, 64]):
        self._data = data
        self.size = config['forest_size']
        self.config= config
        self.prioritized_replay_alpha = 0.6
        self.replay_buffer = PrioritizedReplayBuffer(REPLAY_MEMORY,alpha=self.prioritized_replay_alpha)
        self.trees = []
        self.current_state = []
        self.current_observation = []
        self.time_step = 0
        self.model_path = MODEL_PREFIX_PATH+exp_file_name
        self.batch_size = 32 * (2**(config['depth']))
        self.model_type = model_type
        self.prioritized_replay = True
        self.beta_schedule =None
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.itera_times = itera_times
    @property
    def data(self):
        return self._data

    def build(self):
        """Build forest structure.

        Args:
            data: DataFeature class object.
            tree_amount: int, amount of tree.
        """
        assert isinstance(self.data, DataFeature)
        for index in range(0, self.size):
            # logging.info("-------------[build tree index]: %s"%index)
            tree = Tree.build(self, self.data, index, model_type=self.model_type,
                              model_path=self.model_path, depth=self.config['depth'])
            self.trees.append(tree)

    def setInitState(self,observation):
        """ input observation get from environment, change type to state.

        :param observation: list

        """
        def __set_state(observation):
            # todo
            return list_to_dic(observation)

        self.current_feature = list_to_dic(observation)
        self.current_state = __set_state(observation)
        # np.stack((observation, observation, observation, observation), axis = 2)

    def restore_init_state(self,current_feature,current_state):
        self.current_feature = current_feature
        self.current_state = current_state

    def getInitState(self):
        return (self.current_feature,self.current_state)

    def observation2state(self,observation):
        # np.append(self.current_state[:,:,1:],record['next_observation'],axis = 2)
        return list_to_dic(observation)

    def distribute(self,sample):
        for index,tree in enumerate(self.trees):
            # logging.info("-------------[distribute tree index]: %s"%index)
            tree.distribute(sample)

    def predict(self,for_test=False):
        """
        Attempts to predict the value of the class attribute by aggregating
        the predictions of each tree.
        
        Parameters:
            weighting_formula := a callable that takes a list of trees and
                returns a list of weights.
        """
        assert self.current_state !=[], "None initial current state"
        predictions = {}
        sample = [self.current_state, self.current_feature]
        for tree in self.trees:
            
            _p = tree.predict(sample,for_test=for_test)
            assert type(_p)!=list," predict is a list, it is %s, type %s, impossible as normal!"%(_p,type(_p))
            # if _p is None:
            #     continue
            # if isinstance(_p, CDist):
            #     if _p.mean is None:
            #         continue
            # elif isinstance(_p, DDist):
            #     if not _p.count:
            #         continue
            # predictions[tree] = _p
            max_index = _p
            # max_index = _p.index(max(_p))
            if max_index in predictions:
                predictions[max_index]+=1
            else:
                predictions[max_index]=1
        # if not predictions:
        #     return

        # Normalize weights and aggregate final prediction.
        # weights = self.weighting_method(predictions.keys())
        # if not weights:
        #     return
        # assert sum(weights) == 1.0, "Sum of weights must equal 1."
        # assert (not self.data.is_continuous_class),"this project can not use in continuous class now!"
            # Merge continuous class predictions.
            # total = sum(w*predictions[tree].mean for w, tree in weights)
        # else:
            # Merge discrete class predictions.
        # total = DDist()
        # for weight, tree in weights:
        #     prediction = predictions[tree]
        #     for cls_value, cls_prob in prediction.probs:
        #         total.add(cls_value, cls_prob*weight)

        # return the key with max voting amount.
        # logging.info("predictions is %s selected %s"%(predictions,max(predictions)))
        return max(predictions)

    def set_replay_buffer(self,record):
        """After get reward from environment, Agent should add new record into replay buffer.
        
        Args:
            record: dict type, has following key at least:
                'reward':
                'terminal':
                'next_observation':
        """
        new_state = self.observation2state(record['observation'])
        if type(self.current_state) != dict:
            raise Exception("current state type error")

        if self.prioritized_replay:
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.itera_times
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                           initial_p=self.prioritized_replay_beta0,
                                           final_p=1.0)

        self.replay_buffer.add(self.current_state, record['action'], record['reward'], new_state, float(record['terminal']), record['feature'])
        # self.replayMemory.append([self.current_state,record['action'],record['reward'],new_state,record['terminal'],record['feature']])
        # if len(self.replayMemory) > REPLAY_MEMORY:
        #     self.replayMemory.popleft()
        self.current_state = new_state
        self.current_feature = list_to_dic(record['observation'])

    def update_state(self,record):
        """just update not set record to replay buffer.

        Args:
            record: dict type, has following key at least:
                'reward':
                'terminal':
                'next_observation':
        """
        self.current_state = self.observation2state(record['observation'])        
        self.current_feature = list_to_dic(record['observation'])

    def update_model(self):
        """Update model from experience replay buffer.
        
        """
        # print info
        # state = ""
        # if len(self.replayMemory) <= OBSERVE:
        #     state = "observe"
        # elif len(self.replayMemory) > OBSERVE and self.time_step <= OBSERVE + EXPLORE:
        #     state = "explore"
        # else:
        #     state = "train"

        # print("time_step", self.time_step, "/ STATE", state)


        # Step 1: obtain random minibatch from replay memory
        # logging.info("in sampling")
        if self.prioritized_replay:
            minibatch = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.time_step))
        else:
            assert(False,"error conditions")

        # minibatch = self.replay_buffer.sample(self.batch_size)
        # Step 2: distribute
        # logging.info("in distributing")
        for data in minibatch:
            self.distribute(data)
        # Step 3: update
        # logging.info("in training")
        for tree in self.trees:
            # logging.info("total activated node are %s"%len(tree.activated))
            assert type(tree.activated)== set,"activated type error"
            for activated_node in tree.activated:
                activated_node.train_driver(activated_node.sample_list)
        # logging.info(" training completed")
        # step4 clear state.

        self.clear_activated_node()
        self.time_step += 1


    def update_to_all_model(self):
        if self.time_step > OBSERVE:
            # Step 1: obtain random minibatch from replay memory
            # logging.info("in sampling")
            if self.prioritized_replay:
                minibatch = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule.value(self.time_step))
            else:
                assert(False,"error conditions") 
            for sample in minibatch:
                for tree in self.trees:
                    for leaf in tree.leaves_list:
                    # leaf = tree.leaves_list[0]
                        data = copy.deepcopy(sample)
                        data[0] = leaf.filter_state(sample[0])
                        data[3] = leaf.filter_state(sample[3])
                        sample_list_add_data(leaf.sample_list, data)
                        leaf._tree.activated.add(leaf)
                        
            for tree in self.trees:
                # logging.info("total activated node are %s"%len(tree.activated))

                for activated_node in tree.activated:
                    activated_node.train_driver(activated_node.sample_list)
            # logging.info(" training completed")
            # step4 clear state.
            self.clear_activated_node()

        self.time_step += 1


    def initial_model(self):
        # step1 : distribute data in replay buffer.
        all_data = self.replay_buffer.get_all_data()
        for sample in all_data:
            self.distribute(sample)
        # step2: training all of node.
        for index,tree in enumerate(self.trees):
            # logging.info("-------------[initial model training]: %s"%index)
            tree.initial_model()
        # step3: clear state.
        for index,tree in enumerate(self.trees):
            # logging.info("-------------[initial model clearning]: %s"%index)
            tree.clear_leaf_sample()
            tree._tree.clear_node_sample_number()
        self.clear_activated_node()

    def clear_activated_node(self):
        for tree in self.trees:
            tree.activated = set()

def get_values(data, attr):
    """
    Creates a list of values in the chosen attribut for each record in data,
    prunes out all of the redundant values, and return the list.  
    [todo] can be done just one time then be stored..
    """
    return data.uni_attri_value[attr]

def split_values(data, attr):
    """
    Creates a list of values in the chosen attribute, 
    including a random value in the attribute range of data set and a MAX VALUE
    """
    extr_value = data.extre_attri_value[attr]
    range_value = data.data_range[attr]
    min_gap = 0.01
    # logging.info(extr_value)
    # logging.info(attr)
    min_value = extr_value[0] if extr_value[0] > range_value[0] + min_gap else range_value[0] + min_gap
    max_value = extr_value[1] if extr_value[1] < range_value[1] - min_gap else range_value[1] - min_gap
    # print("split_values [min: %s, max: %s]"%(min_value,max_value))
    random_value = float(random.randint(int(min_value * PRECISION), int(max_value * PRECISION)))
    selected_val = random_value/PRECISION
    return [selected_val, MAX_VALUE]

def random_choose_attribute(data, attributes):
    """
    [change]Choose a attribute just randomly.

    Args:
        data: datafeatrue instance
        attributes: a list of attribute. eg. ['a', 'c', 'b', 'd']
    """
    length = len(attributes)
    attr = random.randint(0,length-1)
    return attributes[attr]

def create_decision_tree(attributes, class_attr, wrapper,current_deep, father_node, **kwargs):
    """
    Returns a new decision tree based on the examples given.
    Args:
        data:
        attributes:
        class_attr:
        fitness_func:
        wrapper: pass Class Tree to record all of parameter in trees.
    """
    assert class_attr not in attributes, "Class attributes should not in class attribute"
    node = None
    if current_deep >= wrapper.max_deep:
        # stop create tree node only if enough deep.
        logging.info(" create model index : %s"%wrapper.leaf_count)
        wrapper.leaf_count += 1
        # 【temp】
        # if wrapper.leaf_count >1:
        #     return node
        node = Node(model_type=wrapper.model_type, tree=wrapper, node_name=MODEL_NAME+'-'+str(wrapper.tree_index)+'-'+str(wrapper.leaf_count))
        node.leaf_attributes = attributes
        node.father_node = father_node
        wrapper.leaves_list.append(node)
        return node
    else:
        #[to change] random select next attribute
        # Choose the next best attribute to best classify our data
        best = random_choose_attribute(
            wrapper.data,
            attributes)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        # tree = {best:{}}
        node = Node(model_type=wrapper.model_type, tree=wrapper, attr_name=best)
        logging.info("random choose attribute is %s"%best)
        # [question] n is data amount in current node. 
        node.leaf_attributes = [attr for attr in attributes if attr != best]
        node.father_node = father_node
        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        # [todo] get value must be stationary instead of depending on the rest of dataset
        is_continuous = wrapper.get_attribute_type(best) == ATTR_TYPE_CONTINUOUS
        if is_continuous:
            # just split into two part randomly 
            values = split_values(wrapper.data,best)
        else:
            values = get_values(wrapper.data,best)
        logging.info("the values of this best attributes:[%s]"%(values))
        for val in values:
            # Create a subtree for the current value under the "best" field
            # [question] for countinus attribute, should we regard it as discrete points. means create a node for each points?
            logging.info("now we choose value (%s),from [%s]"%(val, best))
            #[delete data]
            #  if is_continuous:
            #     selected_data = [r for r in data if r[best] <= val]
            # else:
            #     selected_data = [r for r in data if r[best] == val]
            subtree = create_decision_tree(
                #[delete data] selected_data,
                node.leaf_attributes,
                class_attr,
                wrapper=wrapper,
                current_deep=current_deep+1,
                father_node=node)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            if isinstance(subtree, Node):
                node._branches[val] = subtree
            # elif isinstance(subtree, (CDist, DDist)):
            #     # logging.info("in leaf loop, val is %s"%val)
            #     node.set_leaf_dist(attr_value=val, dist=subtree)
                # [change] add node to leaves list.
            else:
                #[temp]
                # print("skip node")
                raise Exception("Unknown subtree type: %s" % (type(subtree),))
    return node


class Tree(object):
    """
    Represents a single grown or built decision tree.
    """
    
    def __init__(self, forest, data, model_type, *args, **kwargs):
        """Init method just set some parameter of decision tree.
        
        Args:
            data:
            kwargs:
                {
                    "metric":
                    "splitting_n":
                    "auto_grow":
                    "leaf_threshold":
                }
        """
        assert isinstance(data, DataFeature)
        self._data = data
        self.forest = forest
        # Root splitting node.
        # This can be traversed via [name1][value1][name2][value2]...
        self._tree = Node(self, model_type=kwargs.get('model_type', [376, 64]), is_root=True)

        # The total number of leaf nodes.
        self.leaf_count = 0

        # the list of leaf nodes.
        self.leaves_list = []
        
        # The total number of samples trained on.
        self.sample_count = 0
        # store the reference of activated leaves node
        self.activated = set()
        self.forest = forest
        self.model_type = model_type
        # The mean absolute error.
        # self.mae = CDist()
        # self._mae_clean = True

        #[q] what is metric

        # Set the metric used to calculate the information gain
        # after an attribute split.
        # if self.data.is_continuous_class:
        #     self.metric = kwargs.get('metric', DEFAULT_CONTINUOUS_METRIC)
        #     assert self.metric in CONTINUOUS_METRICS
        # else:
        #     self.metric = kwargs.get('metric', DEFAULT_DISCRETE_METRIC)
        #     assert self.metric in DISCRETE_METRICS
        # [todo] prohibit this method, we should not splitting any nodes.
        # Set metric to splitting nodes after a sample threshold has been met.
        # self.splitting_n = kwargs.get('splitting_n', 100)
        
        # Declare the policy for handling missing values for each attribute.
        # self.missing_value_policy = {}
        # [todo]: automatically grow???
        # Allow the tree to automatically grow and split after an update().
        # self.auto_grow = kwargs.get('auto_grow', False)
        

        # Determine the threshold at which further splitting is unnecessary
        # solution: if enough accuracy has been achieved.
        # [q]
        # if self.data.is_continuous_class:
        #     # Zero variance is the default continuous stopping criteria.
        #     self.leaf_threshold = kwargs.get('leaf_threshold', 0.0)
        # else:
        #     # A 100% probability is the default discrete stopping criteria.
        #     self.leaf_threshold = kwargs.get('leaf_threshold', 1.0)
            

        ### Used for forests.
        
        # The prediction accuracy on held-out samples.
        # self.out_of_bag_accuracy = CDist()
        
        # Samples not given to the tree for training with which the
        # out-of-bag accuracy is calculated from.
        # self._out_of_bag_samples = []
        
        # The mean absolute error for predictions on out-of-bag samples.
        # self._out_of_bag_mae = CDist()
        # self._out_of_bag_mae_clean = True

    def __getitem__(self, attr_name):
        return self.tree[attr_name]

    @property
    def data(self):
        return self._data

    # @property
    # def is_continuous_class(self):
    #     return self.data.is_continuous_class

    @property
    def tree(self):
        return self._tree

    def get_attribute_type(self,name):
        return self.data.get_attribute_type(name)

    @property
    def tree_index(self):
        return self._tree_index

    @classmethod
    def build(cls, forest, data, index, model_type, *args,**kwargs):
        """
        Constructs a classification or regression tree in a single batch by
        analyzing the given data.
        Args:
            cls: class method decorator will pass Class object as the first parameter to function imply
            data: 'DataFeature' type parameter, containing data to be builded.
        """
        assert isinstance(data, DataFeature)
        # select fitness_function.
        # if data.is_continuous_class:
        #     fitness_func = gain_variance
        # else:
        #     fitness_func = get_gain
        
        t = cls(forest=forest, data=data, model_type=model_type, *args, **kwargs)
        t._tree_index = index
        t.model_path = kwargs['model_path']
        t.depth = kwargs['depth']
        t.max_deep = len(data.attribute_names)/2 if len(data.attribute_names)/2 <= t.depth else t.depth
        # [todo] sample_count 
        t.sample_count = len(data)
        t._tree = create_decision_tree(
            #[delete data] data=data,
            attributes=data.attri_to_select,
            class_attr=data.class_attribute_name,
            # fitness_func=fitness_func,
            wrapper=t,
            current_deep=0,
            father_node=None
        )
        return t

    def distribute(self, sample):
        copy_sample = copy.deepcopy(sample)
        return self._tree.distribute(copy_sample)

    def predict(self,sample,for_test=False):
        # [temp]
        # return self.leaves_list[0].predict(sample)
        return self._tree.predict(sample,for_test=for_test)

    def _find_data(self,node, attrs):
        """Find data which is the nearest to node.
        Args:
            node: tree node.
            attrs: attribute you want to get.[replace with filter function]
        Returns:
            selected_sample_list
            label_list
        """
        selected_sample_list = sample_list_reset()
        if node.n == 0:
            fn = node.father_node
            sample_list = self._find_data(fn,attrs)
            sample_list_add_sample_list(selected_sample_list, sample_list)
            # selected_sample_list.extend(sample)
        else:
            if node.attr_name == None:
                # this is a leaf node with records.
                sample_list_add_sample_list(selected_sample_list, node.sample_list)
                # for item in leaf_data:
                #     # [todo] filter attribute
                    # sample_list_add_data(selected_sample_list, item)
            else:
                # this is not a leaf node, but some of its children have records.
                for key,children in node._branches.items():
                    if children.n == 0:
                        continue
                    sample_list = self._find_data(children,attrs)
                    sample_list_add_sample_list(selected_sample_list, sample_list)
                    # selected_sample_list.extend(sample)
        return selected_sample_list

    def initial_model(self):
        """Initial nerual network after contructure trees.

        """
        logging.info("[in initial_model...]")
        non_data_index = []
        for index, leaf in enumerate(self.leaves_list):
            logging.error("training %s all %s" % (index, len(self.leaves_list)))
            attrs = leaf.leaf_attributes
            selected_sample_list = self._find_data(leaf, attrs)
            # logging.info("the leaf's attributes is %s"%json.dumps(attrs,indent=2))
            # logging.info("the leaf's record is  %s"%json.dumps(leaf_data,indent=2))

            # if leaf.n == 0:
            #     logging.info("[empty node]")
            #     fn = leaf.father_node
            #     for key,other_leaf in fn._branches.items():
            #         other_leaf_data = other_leaf.sample_list
            #         if other_leaf_data != []:
            #             for item in other_leaf_data:
            #                 selected_sample_list.append([value for key,value in item.items() if key in attrs])
            # else:
            #     logging.info("[not empty node]")
            #     for item in leaf_data:
            #         selected_sample_list.append([value for key,value in item.items() if key in attrs])
            # logging.info("features : %s"%selected_sample_list)
            leaf.train_driver(selected_sample_list)
            # leaf.base_model.trainQNetwork(selected_sample_list)
        for leaf in self.leaves_list:
            leaf.sample_list = sample_list_reset()

    def clear_leaf_sample(self):
        for leaf in self.leaves_list:
            leaf.sample_list = sample_list_reset()



class Node(object):
    """
    Represents a specific split or branch in the tree.
    """
    
    def __init__(self, tree, model_type, attr_name=None,node_name=None,is_root=False):
        
        # The number of samples this node has been trained on.
        self.n = 0
        
        # A reference to the container tree instance.
        self._tree = tree
        
        # The splitting attribute at this node.
        self.attr_name = attr_name

        self.sample_list = sample_list_reset()
        # {attr_name:{attr_value:CDist(variance)}}
        # [delete] self._attr_value_cdist = defaultdict(_get_dd_cdist)
        # self._class_cdist = CDist()
        self.leaf_attributes = []
        self._branches = {} # {v:Node}
        self.father_node = None
        # self.base_model = MLPClassifier(hidden_layer_sizes=(40,), max_iter=70, alpha=1e-4,
        #             solver='sgd', verbose=False, tol=1e-4, random_state=1,
        #             learning_rate_init=.2,learning_rate='adaptive', warm_start=True)
        if self.attr_name == None and not is_root:
            assert node_name != None, "not defined node name"
            self.agent_name = node_name
            # [todo] now assump that all of node in the same deep
            actions = self._tree.data.actions # len(self._tree.data.uni_class_value) 
            # todo: add "- self._tree.max_deep" in the future work
            observations =self._tree.data.observations # - self._tree.max_deep
            # logging.info("c: %s, f: %s"%(class_amount,feature_amount))
            # [todo] now assump
            # self.base_model  = DqnAgent(actions,observations,DqnAgent.SIMPLE_OB, node_name)
            self.session = U.MultiSession(self.agent_name)
            self.path = self._tree.model_path+self.agent_name
            g = tf.Graph()
            with g.as_default():
                with tf.variable_scope(self.agent_name):
                    self.session.make_session(g,CPU_NUM)
                    # [todo] just suit to one dim observation now.
                    self.act, self.train, self.update_target, self.debug = deepq.build_train(
                        session=self.session,
                        make_obs_ph=lambda name: U.BatchInput((observations,), name=name),
                        q_func=mlp(model_type),
                        num_actions=actions,
                        optimizer=tf.train.AdamOptimizer(learning_rate=25e-5),
                        gamma=0.99
                    )
                    exploration_fraction = 0.1
                    self.exploration = LinearSchedule(schedule_timesteps=exploration_fraction*self.tree.forest.itera_times, initial_p=1.0, final_p=FINAL_EPSILON)
                    self.t_val = tf.Variable(0)
                    self.session.initialize()
                    self.session.init_saver()
                    self.update_target()
                    self.session.load_state(self.path)
                    self.time_step = self.session.sess.run(self.t_val)
        else:
            self.base_model = None
    
    # def __getitem__(self, attr_name):
    #     assert attr_name == self.attr_name
    #     branches = self._branches.copy()
    #     for value in self.get_values(attr_name):
    #         if value in branches:
    #             continue
    #         elif self.tree.data.is_continuous_class:
    #             branches[value] = self._attr_value_cdist[self.attr_name][value].copy()
    #         else:
    #             branches[value] = self.get_value_ddist(self.attr_name, value)
    #     return branches

    @property
    def tree(self):
        return self._tree

    def _get_attribute_value_for_node(self, record):
        """
        Gets the closest value for the current node's attribute matching the
        given record.
        """
        
        # Abort if this node is a leaf node. 
        if self.attr_name is None:
            return
        
        # Otherwise, lookup the attribute value for this node in the
        # given record.
        attr = self.attr_name
        attr_value = record[attr]
        
        # get kind of values in this node.
        # [todo] consider continuous attribute distribute.
        attr_values = self.get_values(attr)
        is_continuous = self._tree.get_attribute_type(attr) == ATTR_TYPE_CONTINUOUS
        
        if is_continuous:
            select_value = attr_values[0]
            for critical_value in attr_values:
                # logging.info("compare value %s and critival %s"%(attr_value,critical_value))
                if critical_value > attr_value:
                    select_value = critical_value
                    return select_value
            raise Exception("can not find a valid critical value!!")
        else:
            assert attr_value in attr_values," attribute value of this record not exist in node's key."
            return attr_value    

    def get_values(self, attr_name):
        """
        Retrieves the unique set of values seen for the given attribute
        at this node.
        """
        ret = list(self._branches.keys())
        assert len(set(ret))==len(ret),"repeated key found"
        return ret

    def distribute(self, sample, depth=0):
        """
        Returns the estimated value of the class attribute for the given
        record.

        Args: sample:a dict include current_state, action, reward, terminal, next_state, observation 
        """
        feature = sample[-3] # the -3 element of sample is feature.
        # Lookup attribute value.
        attr_value = self._get_attribute_value_for_node(feature)
        # logging.info("attribute:[%s]"%self.attr_name)
        # logging.info("attribute value:[%s]"%attr_value)
        self.n += 1
        if self.attr_name == None:
            # arrived at leaf node
            sample[0] = self.filter_state(sample[0])
            sample[3] = self.filter_state(sample[3])
            sample_list_add_data(self.sample_list, sample)
            self._tree.activated.add(self)
            # logging.info("[in leaf node], record: %s"%self.sample_list)sample_list
            # logging.info("self-attributes:%s"%self.leaf_attributes)
            # logging.info("[leave this node]")
        else:
            assert attr_value in self._branches,"find attribute value not in any branch when distribute."
            # elif attr_value in self._branches:
            # try:
                # Propagate decision to leaf node.
                # assert self.attr_name
            # logging.info("[to next deep]")
            self._branches[attr_value].distribute(sample, depth=depth+1)
            # [delete] this try-catch not exist.
            # except NodeNotReadyToPredict:
            #     # TODO:allow re-raise if user doesn't want an intermediate prediction?
            #     pass

    def predict(self, sample, depth=0, for_test=False):
        """
        Returns the estimated value of the class attribute for the given
        record.

        Args: sample dict
        """
        feature = sample[-1] # the last element of sample is feature.
        attr_value = self._get_attribute_value_for_node(feature)
        # logging.info("attribute:[%s]"%self.attr_name)
        # logging.info("attribute value:[%s]"%attr_value)
        if self.attr_name == None:
            # arrived at leaf node
            # todo filter attribute function
            # sample = [value for key,value in record.items() if key in self.leaf_attributes]
            state = self.filter_state(sample[0])
            state = np.array(state)# the first element is 'state'
            # logging.info("in predicting ,sample is %s"%sample)
            # assert (not self.is_continuous_class),"this project can not use in continuous class now!"
            if not for_test:
                action = self.act(state[None], update_eps=self.exploration.value(self.time_step))[0]
            else:
                action = self.act(state[None], update_eps=FINAL_EPSILON)[0]
            # print("predict Action is %s"%action)
            # result = self.base_model.getAction(state)
            return action
        else:
            # raise Exception("impossible in single node predict test!")
            assert attr_value in self._branches,"find attribute value not in any branch when distribute."
            # elif attr_value in self._branches:
            # try:
                # Propagate decision to leaf node.
                # assert self.attr_name
            # logging.info("[to next deep]")
            return self._branches[attr_value].predict(sample, depth=depth+1, for_test=for_test)



    def train_driver(self,sample_list):
        obses_t, actions, rewards, obses_tp1, dones = sample_list['obses_t'], sample_list['actions'], sample_list['rewards'], sample_list['obses_tp1'], sample_list['dones']
        weights,indexs = sample_list['weights'],sample_list['indexs']
        # total distribute completed, before train.
        np.array(obses_t)
        np.array(actions)
        np.array(rewards)
        np.array(obses_tp1)
        np.array(dones)
        td_error = self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
        # self.base_model.trainQNetwork(self.sample_list)
        self.sample_list = sample_list_reset()
        self.time_step += 1      
        if self.time_step % UPDATE_TARGET_INTERVAL == 0:
            print("update target times %s, agent name %s"%(self.time_step,self.agent_name))
            self.update_target()
        if self.time_step % SAVE_MODEL_INTERVAL == 0:
            # pass
            print("save model to %s"%self.path)
            self.session.sess.run(tf.assign(self.t_val,self.time_step))
            self.session.save_state(self.path, self.time_step)
        new_priorities = np.abs(td_error) + self.tree.forest.prioritized_replay_eps
        self.tree.forest.replay_buffer.update_priorities(indexs, new_priorities)
        return td_error

    def clear_node_sample_number(self):
        self.n = 0
        if self.attr_name == None:
            return
        for key,children in self._branches.items():
            children.clear_node_sample_number()

    def filter_state(self,state):
        return dic_to_list(state)


def sample_list_reset():
    sample_list = {
        'obses_t':[],
        'actions':[],
        'rewards':[],
        'obses_tp1':[],
        'dones':[],
        'weights':[],
        'indexs':[]
    }
    return sample_list

def sample_list_add_data(sample_list,data):
    obs_t, action, reward, obs_tp1, done, features,weight, index = tuple(data)
    sample_list['obses_t'].append(np.array(obs_t, copy=False))
    sample_list['actions'].append(np.array(action, copy=False))
    sample_list['rewards'].append(reward)
    sample_list['obses_tp1'].append(np.array(obs_tp1, copy=False))
    sample_list['dones'].append(done)
    sample_list['weights'].append(weight)
    sample_list['indexs'].append(index)

def sample_list_add_sample_list(sample_list_1, sample_list_to_copy):
    sample_list_2 = copy.deepcopy(sample_list_to_copy)
    sample_list_1['obses_t'].extend(sample_list_2['obses_t'])
    sample_list_1['actions'].extend(sample_list_2['actions'])
    sample_list_1['rewards'].extend(sample_list_2['rewards'])
    sample_list_1['obses_tp1'].extend(sample_list_2['obses_tp1'])
    sample_list_1['dones'].extend(sample_list_2['dones'])
    sample_list_1['weights'].extend(sample_list_2['weights'])
    sample_list_1['indexs'].extend(sample_list_2['indexs'])