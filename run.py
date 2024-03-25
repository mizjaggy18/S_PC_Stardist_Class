# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


import sys
import numpy as np
# from pathlib import Path
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob

from tifffile import imread
from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, User, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

from csbdeep.utils import Path, normalize
from stardist.models import StarDist2D

import torch
from torchvision.models import DenseNet
import openvino as ov

import tensorflow as tf
from tensorflow import keras

from PIL import Image
from skimage import io, color, filters

# import matplotlib.pyplot as plt
import time
import cv2
import math
import csv

from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.1"
# Stardist (from Cytomine) followed by Tensorflow classification for Pancreatic Cancer (Date created: 9 Jan 2024)

def run(cyto_job, parameters):
    logging.info("----- PC-Stardist-class-DenseNet with OpenVino v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project
    # roi_type=parameters.cytomine_roi_type
    # modeltype=parameters.cytomine_model
    area_th=parameters.cytomine_area_th
    stardist_model=parameters.stardist_model

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    if stardist_model==1:
        modelsegment = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')
    elif stardist_model==2:
        modelsegment = StarDist2D(None, name='2D_versatile_fluo', basedir='/models/')

    # ----- load network ----
    # modelname = "/models/pc-cb-2class.h5"
    # print("Model name: ", modelname)
    # model = tf.keras.models.load_model(modelname, compile=False) 
    # model.compile(optimizer='adam',
    #           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #           metrics=['accuracy'])
    # model.summary()

    # Paths where ONNX and OpenVINO IR models will be stored.
    # ir_path = weights_path.with_suffix(".xml")
    ir_path = "/models/pc-cb-2class_dn21adam_best_model_100ep.xml"

    # Instantiate OpenVINO Core
    core = ov.Core()

    # Read model to OpenVINO Runtime
    #model_ir = core.read_model(model=ir_path)

    # Load model on device
    compiled_model = core.compile_model(model=ir_path, device_name='CPU')
    output_layer = compiled_model.output(0)
    
    # ------------------------

    print("Model successfully loaded!")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded!")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")

        f.write("AnnotationID;ImageID;ProjectID;JobID;TermID;UserID;Area;Perimeter;Hue;Value;WKT \n")
        
        #Go over images
        for id_image in list_imgs2:

            print('Current image:', id_image)
            imageinfo=ImageInstance(id=id_image,project=parameters.cytomine_id_project)
            imageinfo.fetch()
            calibration_factor=imageinfo.resolution
            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_roi_term],
                project=parameters.cytomine_id_project,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            print(roi_annotations)

            if stardist_model==0:
                pass
            else:
                #Go over ROI in this image
                #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
                for roi in roi_annotations:
                    try:
                        #Get Cytomine ROI coordinates for remapping to whole-slide
                        #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                        print("----------------------------ROI------------------------------")
                        roi_geometry = wkt.loads(roi.location)
                        print("ROI Geometry from Shapely: {}".format(roi_geometry))
                        print("ROI Bounds")
                        print(roi_geometry.bounds)
                        min_x=roi_geometry.bounds[0]
                        min_y=roi_geometry.bounds[1]
                        max_x=roi_geometry.bounds[2]
                        max_y=roi_geometry.bounds[3]
                        #Dump ROI image into local PNG file
                        roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                        roi_png_filename=os.path.join(roi_path+'/'+str(roi.id)+'.png')
                        print("roi_png_filename: %s" %roi_png_filename)
                        is_algo = User().fetch(roi.user).algo
                        roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=not is_algo)
                        #roi.dump(dest_pattern=os.path.join(roi_path,"{id}.png"), mask=True, alpha=True)
    
                        #Stardist works with TIFF images without alpha channel, flattening PNG alpha mask to TIFF RGB
                        im=Image.open(roi_png_filename)
    
                        bg = Image.new("RGB", im.size, (255,255,255))
                        bg.paste(im,mask=im.split()[3])
                        roi_tif_filename=os.path.join(roi_path+'/'+str(roi.id)+'.tif')
                        bg.save(roi_tif_filename,quality=100)
                        X_files = sorted(glob(roi_path+'/'+str(roi.id)+'*.tif'))
                        X = list(map(imread,X_files))
                        n_channel = 3 if X[0].ndim == 3 else X[0].shape[-1]
                        axis_norm = (0,1)   # normalize channels independently  (0,1,2) normalize channels jointly
                        if n_channel > 1:
                            print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    
                        #Going over ROI images in ROI directory (in our case: one ROI per directory)
                        for x in range(0,len(X)):
                            print("------------------- Processing ROI file %d: %s" %(x,roi_tif_filename))
                            if stardist_model==1:
                                X1=X[x]
                            elif stardist_model==2:
                                #Preprocessing for PR-IHC
                                X2=X[x]
                                blurred_image = filters.gaussian(color.rgb2gray(X2), sigma=1.0)      
                                mask = blurred_image > 0.8  # Adjust the threshold as needed                            
                                # Use the mask to remove the background
                                X2[mask] = [255, 255, 255]  # Set background pixels to black (0, 0, 0)                            
                                X1=255 - X2[:,:,1]
    
    
                            img = normalize(X1, parameters.stardist_norm_perc_low, parameters.stardist_norm_perc_high, axis=axis_norm)
                            n_tiles = modelsegment._guess_n_tiles(img)
                            #Stardist model prediction with thresholds
                            labels, details = modelsegment.predict_instances(img,
                                                                      prob_thresh=parameters.stardist_prob_t,
                                                                      nms_thresh=parameters.stardist_nms_t,
                                                                      n_tiles=n_tiles)
                            print("Number of detected polygons: %d" %len(details['coord']))
                            cytomine_annotations = AnnotationCollection()
                            #Go over detections in this ROI, convert and upload to Cytomine
                            for pos,polygroup in enumerate(details['coord'],start=1):
                                #Converting to Shapely annotation
                                points = list()
                                for i in range(len(polygroup[0])):
                                    #Cytomine cartesian coordinate system, (0,0) is bottom left corner
                                    #Mapping Stardist polygon detection coordinates to Cytomine ROI in whole slide image
                                    x_ratio = (max_x-min_x)/im.size[0]
                                    y_ratio = (max_y-min_y)/im.size[1]
                                    p = Point(min_x+(polygroup[1][i]*x_ratio),max_y-(polygroup[0][i]*y_ratio))
                                    points.append(p)
    
                                annotation = Polygon(points) # coordinates denoting pixels
                                area=annotation.area * (calibration_factor ** 2) # to convert annotation.area (in pixels) to micron2                
                                if area > area_th: 
                                    #Append to Annotation collection 
                                    cytomine_annotations.append(Annotation(location=annotation.wkt,
                                                                           id_image=id_image,#conn.parameters.cytomine_id_image,
                                                                           id_project=parameters.cytomine_id_project,
                                                                           id_terms=[parameters.cytomine_id_cell_term]))
                                    print(".",end = '',flush=True)
    
                            #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                            cytomine_annotations.save()
                    except:
                        print("An exception occurred. Proceed with next annotations")

            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_cell_term],
                project=project.id,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            # print(roi_annotations)

            job.update(status=Job.RUNNING, progress=60, statusComment="Running classification on image...")

            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0            
            
            roi_numel=len(roi_annotations)
            x=range(1,roi_numel)
            increment=np.multiply(10000,x)

            #Go over ROI (nuclei) in this image
            for i, roi in enumerate(roi_annotations):
                
                for inc in increment:
                    if i==inc:
                        shutil.rmtree(roi_path, ignore_errors=True)
                        import gc
                        gc.collect()
                        print("i==", inc)

                roi_geometry = wkt.loads(roi.location)

                # print("----------------------------Cells------------------------------")
                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/')
                roi_png_filename=os.path.join(roi_path+str(roi.id)+'.png')

                is_algo = User().fetch(roi.user).algo
                roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=not is_algo)
                    
                roi.delete() #delete stardist annotation to avoid double annotations
                
                
                im = cv2.cvtColor(cv2.imread(roi_png_filename),cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(224,224))
                arr_out = im.reshape(-1,224,224,3)
                output = np.zeros((0,num_classes))
                arr_out = torch.from_numpy(arr_out.transpose(0, 3, 1, 2))                
                output_batch = compiled_model([arr_out])[output_layer]
                output = np.append(output, output_batch, axis=0)
                pred_labels = np.argmax(output, axis=1)
                # pred_labels=[pred_labels]
                pred_all.append(pred_labels)

                if pred_labels[0]==0:
                    # print("Class 0: Tumor")
                    id_terms=parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1
                elif pred_labels[0]==1:
                    # print("Class 1: Normal")
                    id_terms=parameters.cytomine_id_c1_term
                    pred_c1=pred_c1+1
                
                cytomine_annotations = AnnotationCollection()
                annotation=roi_geometry
                
                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                       id_image=id_image,#conn.parameters.cytomine_id_image,
                                                       id_project=project.id,
                                                       id_terms=[id_terms]))
                print(".",end = '',flush=True)


                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()

            end_prediction_time=time.time()

            job.update(status=Job.RUNNING, progress=90, statusComment="Generating scoring for whole-slide image(s)...")
            pred_all=[pred_c0, pred_c1]            
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1
            print("pred_total:",pred_total)
            print("pred_tumor:",pred_c0)
            print("pred_normal:",pred_c1)
                  
            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            print("Prediction time: ",end_prediction_time-start_prediction_time)

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0 (Tumor);Class 1 (Normal);Total Prediction;Execution Time;Prediction Time\n")
            f.write("{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_total,end_time-start_time,end_prediction_time-start_prediction_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

