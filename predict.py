#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class covid:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('covid19_model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Normal'
            return [prediction]
        if result[0][1] == 1:
            predtiction = "Covid-19"
            return [predtiction]
        else:
            prediction = 'Viral Pneumonia'
            return [ prediction]


