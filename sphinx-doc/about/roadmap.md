# Road Map


## Done


- U-Net Training / Inference
- FCN Classification Training / Inference
- Extract Segmentation Maps from markers drawn with a specific color
- Convert segmentation maps in between various formats (polygons, label-studio, binary, coco, ...)
- Map segmentation maps to images by name (e.g. segmentation maps are named `<image_name>_label.png`)
- Extract Class information from folder structure
- Load image paths via regex (optional recursively)
- Load image paths with native folder browser
- Calculate Image quality with various scores, e.g. brisque
- Image Registration via pyelastix
- Halcon-style Variatonal Model
- Skeletonization of binary images with voronoi or openCV
- Normalize image with different methods, e.g. opencv CLAHE or local histogram normalization
- Extensive batch-wise tiling, resizing, padding, stitching
- Morphologically process images
- Run interactive Watershed with segmentation maps
- Design workflows (memorized module sequences and settings)
- Evaluate Segmentation results (contour objects, segmentation maps)
- Allow Export images
- Upload and automatically extract .zip files to project directory
- Handle project and workspace scopes
- Activity logging
- Multiple Architecture Selection for Segmentation Models
- Run any ImageJ script in a batch
- Deep Reconstruction Models
- ...

## Todo

- Documentation
- Testing
- Bug-Fixing