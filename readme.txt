
Our code can be downloaded from GitHub:  
  https://github.com/jlm-138/DA_YOLO.git



-----Weight Download
  Link: https://pan.baidu.com/s/1nD1-uRPK24j5Zhxo8JnGQg
  Password: ofo1



-----Dataset Download
Download the VEDAI dataset:
  https://www.kaggle.com/datasets/pepsissalom/vedaidataset
  or 
  https://downloads.greyc.fr/vedai/

Download the DroneVehicle dataset:
  Link：https://pan.baidu.com/s/1O4nD_goHciNhjEhynNOIBQ 
  Password：x5xu
  or
  https://github.com/VisDrone/DroneVehicle?tab=readme-ov-file

-----Training the Model
1. Preparing the Dataset
  This paper uses the VOC format for training, and you need to prepare the dataset before training.
  Before training, place the annotation files of the visible light images in the Annotation folder under VOCdevkit/VOC2007.
  Before training, place the image files of the visible light images in the JPEGImages folder under VOCdevkit/VOC2007.

  Before training, place the annotation files of the infrared images in the Annotation folder under VOCdevkit_ir/VOC2007.
  Before training, place the image files of the infrared images in the JPEGImages folder under VOCdevkit_ir/VOC2007.

2. Processing the Dataset
  After placing the dataset, use voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training,
  and use voc_annotation_ir.py to obtain 2007_train_ir.txt and 2007_val_ir.txt for training.
  Modify the parameters in voc_annotation.py. For the first training, you can only modify the classes_path, which points to the txt file of the detected categories.
  When training your own dataset, you can create a cls_classes.txt file and write the categories you need to distinguish in it.
  The content of the model_data/cls_classes.txt file is as follows:

python
  ""
  cat
  dog
  ...
  ""
  Modify the classes_path in voc_annotation.py to point to cls_classes.txt, and run voc_annotation.py.
  The same applies to voc_annotation_ir.py.

3. Starting Network Training
  The classes_path in train.py points to the txt file of the detected categories, which is the same as the one in voc_annotation.py! You must modify it when training your own dataset!
  After modifying classes_path, you can run train.py to start training. After training for multiple epochs, the weights will be generated in the logs_ir folder.

4. Prediction with Trained Results
  Two files are needed for prediction: yolo.py and predict.py. Modify model_path and classes_path in yolo.py.
  model_path points to the trained weight file in the logs folder.
  classes_path points to the txt file of the detected categories.
  After modification, you can run predict.py for detection. Enter the image path to detect.



-----Prediction Steps
1. Train according to the training steps.
2. In the yolo.py file, modify model_path and classes_path in the following part to correspond to the trained files; model_path corresponds to the weight file in the logs folder, and classes_path is the class corresponding to model_path.
3. Set in predict.py for fps testing and video detection.



-----Evaluating Your Own Dataset
1. This paper uses the VOC format for evaluation.
2. If you have already run voc_annotation.py before training, the code will automatically divide the dataset into training, validation, and test sets. To modify the test set ratio, you can change the trainval_percent in voc_annotation.py. trainval_percent specifies the ratio of (training + validation) to the test set. By default, (training + validation)
= 9:1. train_percent specifies the ratio of training to validation within (training + validation). By default, training
= 9:1.
3. After dividing the test set with voc_annotation.py, go to the get_map.py file and modify classes_path to point to the txt file of the detected categories, which is the same as the one used during training. You must modify it when evaluating your own dataset.
4. Modify model_path and classes_path in yolo.py. model_path points to the trained weight file in the logs folder. classes_path points to the txt file of the detected categories.
5. Run get_map.py to obtain evaluation results, which will be saved in the map_out folder.



-----Reference
Our code is based on modifications made to the repository at https://github.com/bubbliiiing/yolov7-pytorch


