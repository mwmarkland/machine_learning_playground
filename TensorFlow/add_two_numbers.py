# This code started as the example from the following StackOverflow question
# https://stackoverflow.com/questions/40518398/error-when-adding-two-large-numbers-in-tensorflow

import tensorflow as tf

# These appear to create a variable and initialize it to the first value.
# These are scalar values.
var1 = tf.Variable(3, tf.int64,name="var1")
var2 = tf.Variable(5, tf.int64,name="var2")
result = tf.Variable(0,tf.int64,name="result")

add_op = tf.add(var1, var2,name="add_op")

result = tf.assign(result,add_op)

init_op = (tf.global_variables_initializer())

print(tf.get_default_graph().get_operations())

with tf.Session() as sess:
    writer = tf.summary.FileWriter("/tmp/markland/tensorflow",graph=tf.get_default_graph())
    sess.run(init_op)
    sess.run(add_op) # I don't think this is needed. Running the "result" node should force this operation.
    sess.run(result)
    print(result.eval())
