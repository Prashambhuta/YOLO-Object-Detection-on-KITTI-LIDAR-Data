# Yolo on KITTI LiDAR

Object Detection is the task of finding objects within an image or video. The task is
not only to find the object but to label it and create a bounding box around the object. In
autonomous navigation object detection is used to detect cars, pedestrians, bicycles, vans, and
other road objects to perform accurate maneuvering. The critical component for all of these is
data processing, and models to understand the surroundings using the data. There are either
mounted Visual Cameras, Laser Detection & Ranging (LiDAR), or a combination of both.
LiDAR is a sensing technology that uses a laser to measure the distance of objects from
the sensor and to create a map of the environment. 360◦ Velodyne Laserscanner was used for
one of the popular autonomous driving datasets, i.e. the Karlsruhe Institute of Technology
and Toyota Technological Institute (KITTI) dataset. For object detection the study focuses on
the neural network architecture of You Only Look Once (YOLO), particularly YOLO8.

The complexity of the task is to detect
pedestrians, cars, signals, and cyclists, and to make decisions in real time. The [YOLO8](https://github.com/ultralytics/ultralytics) performs exceptionally well on real time data, and is the setting new performance benchmarks for image classification, and instance segmentation tasks.

## Deep learning with LiDAR data

It is possible to use deep learning methods with LiDAR. Deep learning is a type of machine
learning that uses neural networks to learn complex patterns and make predictions based on
large amounts of data. To take advantage of faster, and newer neural network architectures,
the methods have been used to convert the point clouds or a Birds Eye View (BEV).

Some time ago a new model You Only Look Once (YOLO) was proposed, by Redmon,
using an end-to-end neural network that makes predictions of bounding boxes and class
probabilities simultaneously. It differs from the approach taken by previous object detection
algorithms, which repurposed classifiers to perform detection. The key advantages of YOLO
are speed (quick inference in real-time), simple architecture, accuracy, and generalization over
all types of data. In their method Redmond and et. al., use a YOLO4 architecture based on DarkNet Image classification neural network architecture. In the paper by Simon et al. they implement YOLOv2 architecture for object detection in
the KITTI dataset. They introduce the Euler-Region Proposal Network (E-RPN) to estimate
the pose of the network.

## Code
The setup for training is taken from [Maudzung's Complex YOLO4 Pytorch training repository](https://github.com/maudzung/Complex-YOLOv4-Pytorch). To understand how the code runs please visit the repo.

The goal is to use the same training process steps but replace the YOLO architecture with the YOLO8 architecture from ultralytics.

### Current Standings (23 Jan 2024)
The current situation is, the loss function has to be completely rewritten based on the architecture of the YOLO8 model. The model instantisation, and parameters are all up to the order required for the KITTI dataset.

The following issues keep track of training the KITTI dataset using the YOLO8 architecture.

* [Issue 1](https://github.com/ultralytics/ultralytics/issues/1765)
* [Issue 2](https://github.com/ultralytics/ultralytics/issues/1058)

It would be interesting if this issues are picked up and resolved, and to see the development of the YOLO8 model on the KITTI dataset.


## Data

Karlsruhe Institute of Technology and Toyota Technological Institute (KITTI) developed a
dataset for autonomous navigation by using a VW station wagon. It has recorded 6 hours of
traffic scenarios with RGB, Greyscale stereo cameras, a Velodyne 3D laser scanner, and Global
Positioning System(GPS)/Inertial Measurement Unit(IMU) systems. The image shows the  car setup for the KITTI dataset.

![Kitti-dataset-01](/media/vw-setup-1.png)

The data is divided into categories for ’Road’, ’City’, ’Residential’, ’Campus’, and ’Person’.
The images are both color and greyscale and stored using 8-bit PNG files. The images are
edited, and the hood and sky are cropped out. The Velodyne laser data is stored as point
binaries. Each point is stored with its (x,y,z) coordinate and an additional reflectance value
(r). Reflectance value is the return intensity or strength of the signal. The classes defined are
’Car’, ’Pedestrian’, ’Van’, ’Truck’, ’Person’, ’Cyclist’ and ’Misc’. Further insights can be seen
in the figure.

![Kitti-dataset-02](/media/dataset-obj-class.png)
![Kitti-dataset-03](/media/dataset-obj-per-image.png)

The RGB view of the cloud points can be seen:

![Kitti-rgb-view](/media/rgb-bev-view-kitti-data.png)


## Method

An object detector model consists of multiple parts, usually a body and a head.
The body consists of complex convolutional neural nets, while the head is used to predict
classes and bounding boxes.

![two-stage-detector](/media/two-stage-object-detector.png)

The YOLO architecture is made up mostly of alternating layers of convolutions and pooling. The figure below details the architecture for YOLO2 model.

![yolo-architecture](/media/yolo-architecture.png)

## Experiments

The goal is to restructure the training by changing the model architecture from Yolo2 to yolo8. The performance of yolo2 model is compared to other models in the Results section below.

![training-pipeline](/media/yolo-pipeline.png)

The images shows the training pipeline for experimenting, first the birds eye view is generated from the cloud points data. The BEV is then fed into the neural network architecture for training, and the predictions are made on the point clouds.

## Results

We see the results for ComplexYOLO performing against the Lidar and Lidar+Mono methods. The advantage of using the YOLO architecture is performing  real time detection. The second figure shows the mAP vs the FPS for all popular models when tested on a dedicated GPU. The Complex Yolo model performs at higher FPS with similar precision scores to other models.

![yolo2-results-01](/media/complex-yolo-performance-2.png)



![yolo2-results-02](/media/complex-yolo-performance.png)


The point clouds are converted to Birds-Eye-View RGB-map and image classification in performed on the BEV.

![yolo-on-BEV](/media/complex-yolo.png)

## Conclusions

* Object detection on Lidar data is a complex task, and currently is behind the visual methods. If successfully implemented the advantages of such methods, efficiency in any condition (robustness), and highly dense data availability.

![yolo-obj-detection-01](/media/conclusion-1.png)

![yolo-obj-detection-02](/media/conclusion-2.png)

## Limitations & Future scope

* The data preprocessing and conversion of 3D cloud points remains a critical point in object detection.
* Low availability of training data, and difficulty of migrating the existing models to other types of Lidar scanners. The setup requires a special calibration each time, and would be difficult to replicate each time.









## References

* Official website to download the kitti dataset: https://www.cvlibs.net/datasets/kitti/, 2016.
* Yolov8 ultralytics github: https://github.com/ultralytics/ultralytics, 2023.
* Jiajun Deng, Shaoshuai Shi, Peiwei Li, Wengang Zhou, Yanyong Zhang, and Houqiang Li.
Voxel r-cnn: Towards high performance voxel-based 3d object detection. In Proceedings of
the AAAI Conference on Artificial Intelligence, volume 35, pages 1201–1209, 2021.
* Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics:
The kitti dataset. International Journal of Robotics Research, 32(11):1231–1237, 2013.* Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer
vision, pages 1440–1448, 2015.
* Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies
for accurate object detection and semantic segmentation, 2014.* Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. MIT Press, 2016.* Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436–
444, 2015.
* Charles R. Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J. Guibas. Frustum pointnets
for 3d object detection from rgb-d data, 2018.
* Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once:
Unified, real-time object detection, 2016.
* Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time
object detection with region proposal networks, 2016.
* Martin Simon, Stefan Milz, Karl Amende, and Horst-Michael Gross. Complex-yolo:
Real-time 3d object detection on point clouds. arXiv preprint arXiv:1803.06199, 2018.
* Hai Wu, Chenglu Wen, Shaoshuai Shi, Xin Li, and Cheng Wang. Virtual sparse convolution
for multimodal 3d object detection. arXiv preprint arXiv:2303.02314, 2023.
* Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d
object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), June 2018.



