What you get from Using ImageProcessor
========================================


Settings Persistence
________________________________

Preview Functionality
________________________________

Batch processing
________________________________

Activity Logging
________________________________

Parameter Logging
________________________________

Workflow Builder
________________________________

Code Sharing
________________________________



What you get from Using TFModelProcessor
=========================================


Loss functions
________________________________

Available loss functions for segmentation masks are:

- Weighted focal jaccard loss
    For each class, you can specify
    how much it impacts the jaccard loss.
    E.g. set 1. for background and 5. for foreground,
    than the foreground will impact the jaccard loss 5 times more.
- Focal Dice loss
    100 * focal loss + Dice Loss
- Focal Loss
    Simple focal loss
- Dice Loss
    Simple Dice loss
- Categorical Jaccard Loss
    Simple Jaccard loss
- Distance-Weighted Focal Loss (useful for imbalanced segmentation masks)
    Similar like in the U-Net paper, we approximate the distance transform.
    Instead of doing it before training, we do it dynamically during training.
    This way we can also compute the distance transform for the current label mask.
    The network feedback is weighted by the distance transform.
    This way, pixels that are close to a class border are paid more attention.


Tensorboard logging
________________________________

Mavis will start tensorboard and log all trainings for you.
It will log:

- Sample patch of the dataset
- Training configuration
- All metrics and losses selected
- Time, date, project, preset and model name

Keras training
________________________________

Mavis works with keras. To change the model in an
module, you can drop-in your own keras implementations.

tf.Dataset handling
________________________________

Mavis works with tf.data.Datasets.
It will automatically prefetch, batch and split your data.

Data augmentation with `albumentations`
_______________________________________

Currently available data augmentation there are

- RandomCropConfig
- MaskCropConfig
- FlipUDConfig
- FlipLRConfig
- ZoomInConfig
- ZoomOutConfig
- BlurConfig
- AdditiveGaussianNoiseConfig
- ContrastConfig
- SaturationConfig
- BrightnessConfig
- CompressionConfig
- ChangeColorTemperatureConfig
- AffineTransformConfig
- InvertBlendConfig
- ElasticTransformConfig
- CutoutConfig
- ColorJitterConfig
- CenterCropConfig
- HardNegativeCropConfig

Adding new data augmentation is as simple as writing
a streamlit parameter block for a new
albumentations class

Multi-GPU training
________________________________

By default, mavis uses the mirrored strategy on all GPUs available
You can configure the available GPU's for mavis with
the environment flag, e.g. CUDA_VISIBLE_DEVICES=0,1

Befor training, mavis will show the GPU and CPU
memory available and used.


Patch Selection Algorithms
________________________________

There are three patch selections implemented that can be selected in the
augmentation pipeline. The patch selection algorithms are  useful
for mask segmentation.

- Crop where wrong prediction is not emtpy, if exists.
    This will crop patches from the image, where the model is performing bad.
    It is peferred to chose around 20% of the patches as negative examples.
    This technique is also referred to as hard-negative-mining.
- Crop where mask is not empty, if exists.
    This crop patches from the image, where the mask is not black.
    This is especially useful if a lot of the mask is background / black.
    Chose between 70% and 80% of positive crops
    if you have imbalanced classes and the
    model overfits on the background.
- Random Crop.
    Use this at the end of each patch selection pipeline as there might still
    be images that are too big. Random Crop will additionally enforce,
    that the model sees the whole image eventually during training.
- Center Crop.
    Use this after the rotation augmentation to remove unwanted border artefacts.

Sigmoid vs. Softmax Training
________________________________

You can chose to have multi-layer output (one-hot-encoded), or train
with sigmoid output (binary encoded single channel images.
This functionality is legacy, as softmax is preferred.

Per Class Recall, Precision, IoU Score
_______________________________________

For each class, the training handler outputs recall, precision and IoU score
as well as overall IoU score and loss during training.

Live predictions during training
_______________________________________

Every x (default 5) epochs, the column "Training Data Segmented"
will be populated in your project table with life predictions.
This way you can monitor what the model is currently learning

Threaded Training
_______________________________________

You can start the training and the training will run in background,
while you can still use mavis


Pretrained Models
_______________________________________

By default, the segmentation and classification models in mavis are pretrained on ImageNet.
You can select whether to freeze the pretrained model or to retrain the whole model

Learning Rate Schedules
_______________________________________


Currently available are

- Cyclical Learning Rate V2
    Exponential decrease of cycles
- Triangular Learning Rate
    Fixed cycles. Can be combined with reduce learning rate on plateau.
    If combined with reduce learning rate on plateau, the reduction will be applied to the max peak in the cycle.
- No Schedule
    Useful to set an initially high learning rate for SGD and work with
    reduce learning rate on plateau.
- Exponential Decay
    Simple exponential decay
- Polynomial Decay
    Simple polynomial decay

Keras Optimizer
_______________________________________

For training we use optimizer from tensorflow.keras and tensorflow-addons.
Currently available optimizers are

- Adam
    Simple ADAM optimizer. Works very fast for small toy problems.
- SGD
    Simple Stochastic Gradient Decent.
    Used with clip norm to avoid exploding gradient.
    Generally works best and out of the box for most use cases.
    Yet, is typically a bit slower but more stable.
- Lookahead Rectified ADAM
    Includes warmup learning rate schedule, a lookahead optimizer (only applys a step if it gets better)
    Also claims to fix a bug in the original Adam implementaiton
- Stochastic Weighted Average SGD
    Muliple SGD steps are agglomerated to a single average SGD step,
    that is weighted by training success.