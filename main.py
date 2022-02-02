import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys



"""
note that in test time, test_num_updates=10 means output is 1(original loss)+10(sghmc_num_updates)+10(num_updates) dimensions;
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
# tf.get_logger().setLevel('ERROR')

from data_generator import DataGenerator
from model import IPML
from tensorflow.python.platform import flags
from copy import deepcopy

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 10, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
## SGHMC Sampler
flags.DEFINE_integer('sghmc_num_burnin', 0, 'number of sghmc gradient updates during training.')
flags.DEFINE_integer('sghmc_num_sample', 5, 'number of sghmc gradient updates during training.')
flags.DEFINE_integer('sghmc_num_updates', 5, 'number of sghmc gradient updates during training.') # must equal to sghmc_num_burnin + sghmc_num_sample
flags.DEFINE_float('epsilon', 3e-2, 'step size of sampler, epsilon ** 2 approx update_lr.') # 
flags.DEFINE_float('mdecay', 1.0, 'sampler hyper.') # 
flags.DEFINE_float('prior_constant', 1e-4, 'prior hyper.') # 
## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', True, 'if True, do not use second derivatives in meta-optimization (as ipml does need to use)')
flags.DEFINE_bool('clip_z_grad', True, 'if True, clip grad in sghmc step')
flags.DEFINE_bool('clip_maml_grad', False, 'if True, clip grad in maml step')
flags.DEFINE_bool('get_z_samples', False, 'if True, get_z_samples')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

flags.DEFINE_integer('num_tasks', 20, 'number of batch tasks in total.')
flags.DEFINE_integer('num_select', 10, 'number of batch tasks selected in active task selection.')
flags.DEFINE_string('mode', 'MILT', 'criterion')
flags.DEFINE_bool('backward', True, 'if false, no model_backward.')

def active_task_selection_and_train(model, model_backward, saver, sess, exp_string, data_generator, mode='MILT', resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = SUMMARY_INTERVAL
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = SUMMARY_INTERVAL
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*2

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        
    print('====> Done initializing, start task selection. Total {} tasks, {} to select.'.format(FLAGS.num_tasks,FLAGS.num_select))
    print('====> Selection criterion: ' + mode)
    if mode == 'MILT':
        """ First training the model_backward """
        print('----> Start training the backward model.')
        num_classes = data_generator.num_classes # for classification, 1 otherwise
        for itr in range(FLAGS.num_tasks):
            feed_dict = {}
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.data_pool[itr]
                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]
            else:
                batch_x, batch_y = data_generator.data_pool[itr]
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = inputa # for training the backward model only
            labelb = labela
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                        model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}
                
            input_tensors = [model_backward.learning_op] # perform (continue) learning
            result = sess.run(input_tensors, feed_dict)
        print('----> Finish training the backward model.')
        """ Start active learning, along with online learning and unlearning for computing the MILT criterion. """
        
        data_generator.selected_data_pool = []
        selected_indices = []
        for itr in range(FLAGS.num_select):
            criterion = []
            for i_s in range(FLAGS.num_tasks):
                if i_s in selected_indices:
                    criterion.append(-999)
                    continue
                feed_dict = {}
                if 'generate' in dir(data_generator):
                    batch_x, batch_y, amp, phase = data_generator.data_pool[i_s]

                    if FLAGS.baseline == 'oracle':
                        batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                        for i in range(FLAGS.meta_batch_size):
                            batch_x[i, :, 1] = amp[i]
                            batch_x[i, :, 2] = phase[i]
                else:
                    batch_x, batch_y = data_generator.data_pool[i_s]
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = inputa
                labelb = labela
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                            model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

                """ for the model """
                input_tensors = [model.Z_samples]
                result = sess.run(input_tensors, feed_dict)
                """ for the backward model """
                input_tensors = [model_backward.unlearning_op]
                sess.run(input_tensors, feed_dict)
                input_tensors = [model_backward.Z_samples]
                result_backward = sess.run(input_tensors, feed_dict)
                input_tensors = [model_backward.learning_op]
                sess.run(input_tensors, feed_dict)
                
                MILT = np.log(np.sum(np.std(result,axis=1)+1e-7)) - np.log(np.sum(np.std(result_backward,axis=1)+1e-7))
                criterion.append(MILT)
                
            print("----> Selecting task. Round ",itr)
            print("Criterion:",criterion)
            max_MILT = max(criterion)
            for i_s in range(FLAGS.num_tasks):
                if criterion[i_s] == max_MILT:
                    data_generator.selected_data_pool.append(data_generator.data_pool[i_s])
                    selected_indices.append(i_s)
                    break
            feed_dict = {}
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.selected_data_pool[-1]
                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]
            else:
                batch_x, batch_y = data_generator.selected_data_pool[-1]
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = inputa
            labelb = labela
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                        model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

            """ for the model """
            input_tensors = [model.learning_op]
            sess.run(input_tensors, feed_dict)
            """ for the backward model """
            input_tensors = [model_backward.unlearning_op]
            sess.run(input_tensors, feed_dict)
    elif mode=='Var':
        """ Start active learning, along with online learning and unlearning for computing the Variance criterion. """
        num_classes = data_generator.num_classes # for classification, 1 otherwise
        data_generator.selected_data_pool = []
        selected_indices = []
        for itr in range(FLAGS.num_select):
            criterion = []
            for i_s in range(FLAGS.num_tasks):
                if i_s in selected_indices:
                    criterion.append(-999)
                    continue
                feed_dict = {}
                if 'generate' in dir(data_generator):
                    batch_x, batch_y, amp, phase = data_generator.data_pool[i_s]

                    if FLAGS.baseline == 'oracle':
                        batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                        for i in range(FLAGS.meta_batch_size):
                            batch_x[i, :, 1] = amp[i]
                            batch_x[i, :, 2] = phase[i]
                else:
                    batch_x, batch_y = data_generator.data_pool[i_s]
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = inputa # b used for testing
                labelb = labela
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                            model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

                """ for the model """
                input_tensors = [model.Z_samples]
                result = sess.run(input_tensors, feed_dict)
                Var = np.sum(np.std(result,axis=1))
                criterion.append(Var)
                
            print("----> Selecting task. Round ",itr)
            print("Criterion:",criterion)
            max_Var = max(criterion)
            for i_s in range(FLAGS.num_tasks):
                if criterion[i_s] == max_Var:
                    data_generator.selected_data_pool.append(data_generator.data_pool[i_s])
                    selected_indices.append(i_s)
                    break
            feed_dict = {}
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.selected_data_pool[-1]
                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]
            else:
                batch_x, batch_y = data_generator.selected_data_pool[-1]
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                        model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

            """ for the model """
            input_tensors = [model.learning_op]
            sess.run(input_tensors, feed_dict)
    elif mode=='ELT':
        """ Start active learning, along with online learning and unlearning for computing the ELT criterion. """
        num_classes = data_generator.num_classes # for classification, 1 otherwise
        data_generator.selected_data_pool = []
        selected_indices = []
        for itr in range(FLAGS.num_select):
            criterion = []
            for i_s in range(FLAGS.num_tasks):
                if i_s in selected_indices:
                    criterion.append(-999)
                    continue
                feed_dict = {}
                if 'generate' in dir(data_generator):
                    batch_x, batch_y, amp, phase = data_generator.data_pool[i_s]

                    if FLAGS.baseline == 'oracle':
                        batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                        for i in range(FLAGS.meta_batch_size):
                            batch_x[i, :, 1] = amp[i]
                            batch_x[i, :, 2] = phase[i]
                else:
                    batch_x, batch_y = data_generator.data_pool[i_s]
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = inputa # b used for testing
                labelb = labela
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                            model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

                """ for the model """
                input_tensors = [model.Z_samples]
                result = sess.run(input_tensors, feed_dict)
                LE = np.log(np.sum(np.std(result,axis=1)+1e-7))
                criterion.append(LE)
                
            print("----> Selecting task. Round ",itr)
            print("Criterion:",criterion)
            max_LE = max(criterion)
            for i_s in range(FLAGS.num_tasks):
                if criterion[i_s] == max_LE:
                    data_generator.selected_data_pool.append(data_generator.data_pool[i_s])
                    selected_indices.append(i_s)
                    break
            feed_dict = {}
            if 'generate' in dir(data_generator):
                batch_x, batch_y, amp, phase = data_generator.selected_data_pool[-1]
                if FLAGS.baseline == 'oracle':
                    batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                    for i in range(FLAGS.meta_batch_size):
                        batch_x[i, :, 1] = amp[i]
                        batch_x[i, :, 2] = phase[i]
            else:
                batch_x, batch_y = data_generator.selected_data_pool[-1]
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = inputa
            labelb = labela
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                        model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

            """ for the model """
            input_tensors = [model.learning_op]
            sess.run(input_tensors, feed_dict)
    elif mode=='Rand':
        data_generator.selected_data_pool = []
        for i_s in range(FLAGS.num_select):
            data_generator.selected_data_pool.append(data_generator.data_pool[i_s])
    else:
        raise ValueError("Unrecognized criterion.")
    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/pre_model' + str(-1))
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    print('====> Done task selection, start training.')
    
    prelosses, zlosses, postlosses = [], [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []
    
    for itr in range(resume_itr, FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.selected_data_pool[itr % FLAGS.num_select]

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]
        else:
            batch_x, batch_y = data_generator.selected_data_pool[itr % FLAGS.num_select]
        inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
        labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
        inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
        labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                        model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, 
                                  model.total_losses2[FLAGS.sghmc_num_updates-1], 
                                  model.total_losses3[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, 
                                      model.total_accuracies2[FLAGS.sghmc_num_updates-1], 
                                      model.total_accuracies3[FLAGS.num_updates-1]])
            if FLAGS.get_z_samples and itr % SAVE_INTERVAL == 0:
                input_tensors.extend([model.Z_samples])
                holder = 1
            else:
                holder = 0

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-3-holder])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            zlosses.append(result[-2-holder])
            postlosses.append(result[-1-holder])

        if itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(zlosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, zlosses, postlosses = [], [], []

        if itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
                    

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if itr % TEST_PRINT_INTERVAL == 0:
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                val_length = len(data_generator.val_data_pool)
                index = np.random.randint(val_length)
                batch_x, batch_y = data_generator.val_data_pool[index]
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb,
                            model_backward.inputa: inputa, model_backward.inputb: inputb,  model_backward.labela: labela, model_backward.labelb: labelb}

                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, 
                        model.metaval_total_accuracies2[FLAGS.sghmc_num_updates-1], 
                        model.metaval_total_accuracies3[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, 
                        model.metaval_total_losses2[FLAGS.sghmc_num_updates-1], 
                        model.metaval_total_losses3[FLAGS.num_updates-1], 
                        model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate()
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, 
                    model.total_accuracies2[FLAGS.sghmc_num_updates-1],
                    model.total_accuracies3[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, 
                    model.total_losses2[FLAGS.sghmc_num_updates-1],
                    model.total_losses3[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]) + ', ' + str(result[2]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    NUM_TEST_POINTS = len(data_generator.val_data_pool)
    NUM_TEST_POINTS = 1
    for _ in range(NUM_TEST_POINTS):
        feed_dict = {}
        if 'generate' not in dir(data_generator):
            batch_x, batch_y = data_generator.val_data_pool[_]
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

        inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
        inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
        labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
        labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]
        if _<=10:
            print(inputa.shape,inputb.shape,labela.shape,labelb.shape)
            print(np.sum(inputa),np.sum(inputb),np.sum(labela),np.sum(labelb))

        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2 + model.metaval_total_accuracies3, feed_dict)
        else:  # this is for sinusoid
            # list_of losses = [model.total_loss1] +  model.total_losses2 +  +  model.total_losses3
            result = sess.run([model.total_loss1] +  model.total_losses2 + model.total_losses3, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    log_path = FLAGS.datasource 
    if FLAGS.num_classes == 20:
        log_path += '20way'
    log_path += '_log/stdout' \
                + '_' + str(FLAGS.update_batch_size) \
                + '_' + str(FLAGS.meta_batch_size) \
                + '_' + str(FLAGS.num_tasks) \
                + '_' + str(FLAGS.num_select) \
                + '_' + FLAGS.mode + '.log'
    logf = open(log_path, 'a')
    sys.stdout = logf
    sys.stderr = logf

    main_seed1 = 11
    random.seed(main_seed1)
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
            # test_num_updates = 5
            print("[Tune]:test_num_updates=", test_num_updates)
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.num_tasks, FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(FLAGS.num_tasks, 1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.num_tasks, FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.num_tasks, FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.num_tasks, FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        input_tensors = None
        metaval_input_tensors = None
    else:
        tf_data_load = False
        input_tensors = None

    # note that here we use a single particle (M=1), where the results in the paper are obtained with M up to 5.
    model = IPML(dim_input, dim_output, test_num_updates=test_num_updates) 
    if FLAGS.backward:
        model_backward = IPML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        # with tf.variable_scope("forward"):
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        # with tf.variable_scope("backward"):
        if FLAGS.backward:
            model_backward.construct_model(input_tensors=input_tensors, prefix='metatrain_',name="R_")
    if tf_data_load:
        # with tf.variable_scope("forward"):
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
        # with tf.variable_scope("backward"):
        if FLAGS.backward:
            model_backward.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_',name="R_")
    model.summ_op = tf.summary.merge_all()
    # model_backward.summ_op = tf.summary.merge_all()
    

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) \
    + '.burnin' + str(FLAGS.sghmc_num_burnin)  \
    + '.sample' + str(FLAGS.sghmc_num_sample)  \
    + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr) \
    + '.seed1' + str(main_seed1) 

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')
    exp_string = exp_string + '_' + str(FLAGS.update_batch_size) \
                + '_' + str(FLAGS.meta_batch_size) \
                + '_' + str(FLAGS.num_tasks) \
                + '_' + str(FLAGS.num_select) \
                + '_' + FLAGS.mode

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        active_task_selection_and_train(model, model_backward, saver, sess, exp_string, data_generator, FLAGS.mode, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()