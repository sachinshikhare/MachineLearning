import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#result = x1 * x2

result = tf.multiply(x1, x2)

print(result)

#session = tf.Session()
#print(session.run(result))
#session.close()

with tf.Session() as session:
	print(session.run(result))