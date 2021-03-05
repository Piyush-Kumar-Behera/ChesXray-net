import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.compat.v1.keras import backend as K 
from tensorflow.keras.preprocessing import image
tf.compat.v1.disable_eager_execution()

def load_image(image_path, preprocess=True, H=320, W=320):
    mean = np.float64(np.loadtxt('metrics/mean.csv'))
    std = np.float64(np.loadtxt('metrics/std.csv'))
    x = image.load_img(img_path, target_size=(H, W))
    print(type(x))
    print(x)
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model,prep_image,image_org,predictions,labels, selected_labels,layer_name='bn'):
    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            alpha=min(0.5, predictions[0][i])
            print(f"Generating gradcam for class {labels[i]}")
            gcam = grad_cam(model, prep_image, i, layer_name)
            gradcam = gcam*255
            gradcam = gradcam.astype(dtype=np.uint8)
            gradcam_jet = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)
            final_image = cv2.addWeighted(image_org,1-alpha,gradcam_jet,alpha,0)
            cv2.imshow(f"{labels[i]}",final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            j += 1

def model_compute(model, image_path, labels, layer_name='bn'):
    prep_image = load_image(image_path)
    image_org = cv2.imread(image_path)
    image_org = cv2.resize(image_org,(320,320))
    predictions = model.predict(prep_image)
    selected_labels = ['Effusion','Mass','Nodule','Cardiomegaly']
    compute_gradcam(model,prep_image,image_org,predictions,labels, selected_labels)

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

base_model = DenseNet121(weights = None, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(labels), activation = "sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights('pretrained_weights/pretrained_model.h5')

img_path = 'images/00011355_002.png'
model_compute(model,img_path,labels)