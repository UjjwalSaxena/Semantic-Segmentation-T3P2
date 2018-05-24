import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob , layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)





def layers( vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    layer7a_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),name='new_layer7a_out')
    # upsample
    layer8a_in1 = tf.layers.conv2d_transpose(layer7a_out, num_classes, 4, 
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), name='new_layer4a_in1')
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer8a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),  name='new_layer4a_in2')
    # skip connection (element-wise addition)
    layer8a_out = tf.add(layer8a_in1, layer8a_in2)
    # upsample
    layer9a_in1 = tf.layers.conv2d_transpose(layer8a_out, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3),name='new_layer3a_in1')
    # 1x1 convolution of vgg layer 3
    layer9a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3), name='new_layer3a_in2')
    # skip connection (element-wise addition)
    layer9a_out = tf.add(layer9a_in1, layer9a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer9a_out, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3) ,name='new_nn_last_layer')
    

    
    return nn_last_layer

tests.test_layers(layers)




def optimize(nn_last_layer, correct_label, learning_rate, num_classes, mode="Train_All"):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer,[-1,num_classes])## size is [-1,num_classes]
    correct_label = tf.reshape(correct_label,[-1,num_classes])
    
    ##Regularization
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01  # Choose an appropriate one.
 
    cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy) #+ reg_constant * tf.reduce_mean(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate) ##using Adam Optimizer
#     training_operation = optimizer.minimize(loss_operation)

    if mode=="Transfer":
      trainable_variables = []
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      for variable in tf.trainable_variables():
        if "new_" in variable.name:
          trainable_variables.append(variable)
      with tf.control_dependencies(update_ops):
          training_operation = optimizer.minimize(cross_entropy, var_list=trainable_variables, name="training_op")
    else:
      training_operation = optimizer.minimize(loss_operation)

    return logits,training_operation,loss_operation

# tests.test_optimize(optimize)





def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, mode="Train_All"):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    if mode=="Transfer":
      custom_initializers = [var.initializer for var in tf.global_variables() if 'new_' in var.name]
      sess.run(custom_initializers)
    else:
      sess.run(tf.global_variables_initializer())
      
    print("Training started..")
    prob= 0.5
    rate= 0.0001
    for i in range(epochs):
        print("Epoch {}".format(i+1))
        for image,label in get_batches_fn(batch_size):
            a,loss= sess.run([train_op,cross_entropy_loss], feed_dict={input_image: image, correct_label: label, keep_prob: prob, learning_rate: rate})
            print("Loss {}".format(loss))  
        print()    
        
    
    
tests.test_train_nn(train_nn)




def run():
    num_classes = 2
    image_shape = (160, 576)
    epochs= 50
    batch_size=10
    mode='Train_All'
#     mode='Transfer'
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    print("checking encoder model")
    helper.maybe_download_pretrained_vgg(data_dir)

    print("model available now")
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        print("get_batches_fn done")
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        correct_label= tf.placeholder(tf.int32,[None,None,None,num_classes], name='correct_label')
        learning_rate= tf.placeholder(tf.float32, name='learning_rate')
        print("variables created")
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out= load_vgg(sess,vgg_path)
        
        print("load_vgg done")
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        print("layers done")
        logits,train_op,cross_entropy_loss= optimize(nn_last_layer, correct_label, learning_rate, num_classes,mode)
        print("optimization done")
        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
        correct_label, keep_prob, learning_rate,mode)
        print("training done")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        print("save_inference_samples done")
        
        # OPTIONAL: Apply the trained model to a video
        builder = tf.saved_model.builder.SavedModelBuilder('./models7')
        builder.add_meta_graph_and_variables(sess,["my-model"])
        builder.save() 
        print("model saved")

if __name__ == '__main__':
    run()
