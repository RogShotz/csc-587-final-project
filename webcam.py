import cv2
import keyboard as kb
import time
import tensorflow as tf
from tensorflow import keras
import os
import PIL.Image
import IPython.display as display
import numpy as np
from keras.models import model_from_json
from timeit import default_timer

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(img)
          loss = calc_loss(img, self.model)

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, img)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img

def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  img_batch = tf.expand_dims(img, axis=0)
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.reduce_sum(losses)

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    display.clear_output(wait=True)
    print ("Step {}, loss {}".format(step, loss))


  result = deprocess(img)
  display.clear_output(wait=True)

  return result

def main():
    model_path = './checkpoints/my_checkpoint.h5'
    model_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)

    vid = cv2.VideoCapture(0) 
    
    model = tf.keras.models.load_model(model_fn)
    deepdream = DeepDream(model)

    start = default_timer()
    while(not kb.is_pressed('f4')):
        print('running')
        ret, frame = vid.read()
        #128, 72
        #256, 144
        #320, 240
        #640, 480
        frame = cv2.resize(frame, dsize=(256, 144))
        #CV2 to PIL
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frame = np.array(frame)

        frame =  run_deep_dream_simple(img=frame, deepdream=deepdream,
                                  steps=20, step_size=.01)
        
        #PIL to CV2
        frame = np.array(frame)
        frame = frame[:, :, ::-1].copy()
        cv2.imshow('frame', cv2.resize(frame, dsize=(640, 480)) )
        duration = default_timer() - start
        print(duration)
        start = default_timer()
        # Break if key detected
        if cv2.waitKey(1) & 0xFF == 240: #make it equal something it cant
            break
    

    vid.release()
    # Destroy all the windows 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()