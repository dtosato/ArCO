# ArCO

## Disclaimer
This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. Feel free to use this code for academic
purposes.  Please use the citation provided below.

The test part of this code takes 0.05 seconds per image on a Intel Xeon(R)
CPU E5440 @2.83 GHz 8 GB RAM on Matlab 2009b. The most computationally expensive part of
this code is the learning phase `Y_train_light.m`

## Citation

Diego Tosato, Michela Farenzena, Mauro Spera, Marco Cristani, Vittorio Murino
“Multi-class Classification on Riemannian Manifolds for Video Surveillance,”
ECCV, 2010.  

## Performance

 This code performs patch-based head pose detection given the a 50x50 image.
 Should you have any problems please email me at diego.tosato@univr.it

## Data

* [QMUL4PoseHeads](https://drive.google.com/open?id=0B0MZ5gr7K36SVFVjYVBpaTFuRFU)

## Demo code and images provides

There is one complete script for learning and testing ARCO:

1. Download data.
2. Z_ARCO.m: this is the main script. It is able to learn and test a
multi-class Logitboost classifier on Riemannian Manifold.
3. The variable 'experiment'  (in Z_ARCO.m) contains a path where all the
pre-computed parts of this framework are stored.
Only the classification results are not computed in order to
show you some qualitative results of this framework.
4. If you want to test this framework on the complete test set, just change
the variable 'test_dir' from './QML4PoseHeads/test_demo' to
'./QML4PoseHeads/test'.
5. If you want to see the statistics of this framework on the complete
test set without run testing, these are in [experiment
'/full_test_results'].

This code is provided with a pre-computed training set and its learned
classifier in order to directly test the classifier.
