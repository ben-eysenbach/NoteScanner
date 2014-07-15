#[NoteScanner](https://github.com/ben-eysenbach/NoteScanner)



###Introduction

The goal of this project is to make an tool to digitize hand-written notes. Taking notes by hand is often easier and faster than typing up notes, especially for technical material. Nonetheless, having a digital copy of your notes is helpful both as a backup and as a quick resource (you can't `grep` over a notebook). There are tools that do this natively (namely [LiveScribe](www.livescribe.com/smartpen/)), but they are significantly more expensive than ordinary pens and paper.

The process of digitizing notes will go something like this:

1. Take a picture of your notes.
2. Use some basic Computer Vision to recognize the sheet of paper and find its corners (hopefully 4).
3. Transform the image so the identified corners are dragged to the corners of the image
4. Recognize handwriting.

=========

###Step 1: Photographing

This is easy. It shouldn't matter if your photograph is greyscale or color (unless you have color drawing in your notes). Using a background with a different color than your notes will make identifying the corners easier.

======


###Step 2: Identifying the Corners

To find the corners, we draw a polygon around the piece of paper, simplify it down to a quadrilateral, and extract the corners. This bounding rectangle is usually referred to as a contour, and the process of finding it is cast as [edge detection](http://en.wikipedia.org/wiki/Edge_detection). We will be using the edge detection algorithm outlined in Suzuki[85]. This algorithm is already implemented in OpenCV as [findContours](http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours).

This algorithm requires that the input image be greyscale. One way to convert an RGB image to greyscale is to sum some/all of the Red-Green-Blue channels. This approach is fast, and requires no prior knowledge about the image. However, we do know that the image will be mainly composed of 2 colors, the paper and the background. Given this, we can use [k-means](http://en.wikipedia.org/wiki/Kmeans) to group pixels into two groups (paper and background). We then color every pixel in the first group black, and every pixel in the second group white. Note, it doesn't matter if color the first group white and the second group black.

Now that we have our grey-scale image (it's actually binary (black and white)), we can run the contour detection algorithm to get a bounding polygon. Even though the paper should appear to be a quadrilateral, some accumulated error will cause the algorithm to actually output a complex polygon, which is close to a quadrilateral. We can reduce the number of points on this polygon using the [Ramer–Douglas–Peucker algorithm](http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm) (implemented as `approxPolyDP` in OpenCV). After checking that we are left with exactly four points in the polygon, we can move onto the next step.

========


###Step 3: Transform Image

This step will "drag" the corners found in step 2 such that the notes will the entire image. Specifically, we don't have a [book scanner](http://www.wired.com/2009/12/diy-book-scanner/), so our image will have some element of perspective in it (i.e. it will not look like a rectangle). The process is known as taking a [Perspective Transform](http://en.wikipedia.org/wiki/3D_projection#Perspective_projection). This process works by taking the corners found in step 2, and finding a function that maps them to the corners of the image. This function is encoded in a 3 x 3 matrix ( $$$ M $$$ ) such that multiplying my a point (given in [homogeneous coordinates](http://en.wikipedia.org/wiki/Homogeneous_coordinates), $$$ \<x, y, 1\> $$$ ) will give the corresponding corner of the image.

========

#####Identity

For example, if our notes occupy the entire image, we would want to map each point on the image to itself. This could be encoded as the identity function ( $$$ f(x) = x $$$ ) or as the identity matrix:

$$ \begin{pmatrix}
1&0&0\\\
0&1&0\\\
0&0&1\\\
\end{pmatrix} $$

-------------

#####Scale

Now imagine that we take a 200 x 200 image of our notes, and the notes occupy exactly the upper-left 100x100 section of the image. In this case, the perspective transform is simply enlarging the original image by a factor of two along both axes (and then cropping). This would be achieved by the function $$$ f(x,y) = (2x, 2y) $$$ or the matrix:
$$
\begin{pmatrix}
2&0&0\\\
0&2&0\\\
0&0&1\\\
\end{pmatrix}
$$.

Generally, to scale an image by $$$ s\_x $$$ in the x direction and $$$ s\_y $$$ in the y direction, map the points on the original image using the matrix:
$$
\begin{pmatrix}
s\_x&0&0\\\
0&s\_y&0\\\
0&0&1\\\
\end{pmatrix}
$$.

------------

#####Rotation

To encode rotation, consider what happens to each [unit vector](http://en.wikipedia.org/wiki/Unit_vector#Cartesian_coordinates) in Cartesian Coordinates when rotated clockwise by a angle $$$ \theta $$$. The $$$ \hat i $$$ vector ( $$$ <1,0> $$$ ) becomes $$$ <\cos(\theta), \sin(\theta)> $$$ and the $$$ \hat j $$$ vector ( $$$ <0,1> $$$ ) becomes $$$ <-\sin(\theta), \cos(\theta)> $$$. Putting this together, we get the 2 x 2 rotation matrix:

$$
\begin{pmatrix}
\cos(\theta)&-\sin(\theta)\\\
\sin(\theta)&\cos(\theta)\\\
\end{pmatrix}\\\
$$

To form the required 3 x 3 matrix, simply paste the 2 x 2 rotation matrix into the upper left of the 3 x 3 identity matrix:

$$
\begin{pmatrix}
\cos(\theta)&-\sin(\theta)&0\\\
\sin(\theta)&\cos(\theta)&0\\\
0&0&1\\\
\end{pmatrix}\\
$$

-------------

#####Translation

So you've probably been wondering why I chose to use vectors of length 3 when only dealing with 2 dimensions. There are 2 parts to my explanation. First, it makes translation easy; we can simply use the matrix:

$$
\begin{pmatrix}
1&0&t\_x\\\
0&1&t\_y\\\
0&0&1\\\
\end{pmatrix}\\
$$

The second explanation is more "mathy." Note that the point mapping function is a linear function which must map 4 points to 4 other points. Each point has 2 element which carry information (the third element only simplifies calculations), for a total of 8 degrees of freedom. Our 3 x 3 transformation matrix has 9 elements, but the 3,3 cell is always 1, so we only get 8 cells of useful information. Yes, math is beautiful.

-------------

#####Combination

Now that we can perform each of these image operations separately, we can combine them by multiplying matrices. For example, let $$$ S $$$ be the scale matrix, $$$ T $$$ be the translation matrix, $$$ R $$$ be the rotation matrix, and $$$ x$$$ be our point. If we want to rotation our image, enlarge it, and then translate it, we can do the following multiplication: $$$( T \cdot (S \cdot (R \cdot x))) $$$. Note that the order in which we apply each operation does matter; enlarging the image and then translating is different from translating and then scaling. However, matrix multiplication is associative, so the above product can be rewritten as $$$ (T \cdot S \cdot R) \cdot x $$$. From this, it is clear that our transformation matrix can be written as a single matrix $$$ M = T \cdot S \cdot R $$$.

With the 3 operations defined above, we can apply many transformations. However, all combinations of these transformations, if applied to a polygon, will result in a [similar](http://en.wikipedia.org/wiki/Similarity_(geometry)) polygon. This is bad news for us, because there are many cases where the notes in the picture do not appear similar to the real shape of the notes (i.e. they may not be a rectangle). We could also consider matrices for reflection, sheering, and squeezing to allow for all affine transformations, but that still wouldn't be enough. Affine transformations require that parallel lines remain parallel, which is nearly not the case in some of the examples here.

------------

#####Linear Algebra

We can overcome this by solving for the transformation matrix directly instead of trying to compose it from individual matrix operations. Specifically, let's make a matrix of corners found in step 2 and solve for the matrix which maps them to the corners of the image (with height  h  and width  w  ).

å
$$
X =
\begin{pmatrix}
x_{ul}&x\_{ur}&x\_{ll}&x\_{lr}\\\
y\_{ul}&y\_{ur}&y\_{ll}&y\_{lr}\\\
\end{pmatrix}
$$

We want to solve for the transformation matrix $$$ M $$$ in the following equation:

$$
M \cdot X =
\begin{pmatrix}
0&w&0&w\\\
0&0&h&h\\\
\end{pmatrix}
= B
$$

The first column of $$$B$$$ is the zero vector and $$$B$$$ is not square, so we cannot find its inverse. Rather, we must use [least squares]("http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Computation"):

$$ M \cdot X = B $$

$$ M \cdot (X \cdot X^T) = B \cdot X^T $$
$$ M \cdot (X \cdot X^T) \cdot (X \cdot X^T)^{-1} = (B \cdot X^T) \cdot (X \cdot X^T)^{-1} $$
$$ M = (B \cdot X^T) \cdot (X \cdot X^T)^{-1} $$

Now, given a point on the original image, we know where it should go on the new image. There are still two major problems. First, we may be told that the point on the new image might not be a integer. Second, there may not be any integer point on the original image that maps to a given point on the new image.

The basic approach to solving both these problems is to go backwards. For every point on the new image, find the corresponding point on the original image. We can do this by finding the inverse of the transformation matrix. This matrix will have an inverse exactly when no 3 corners found in step 2 are collinear.

Consider what it would mean for 3 corners to be collinear. On the new image, these 3 corners form a triangle which occupies half of the new image. All the information for this triangle must come from the region bounded by this "triangle" in the original image. But this "triangle" is a line, so it encloses exactly 0 pixels, so we have no idea of what should go in the triangle in the new image.

By extension, if 3 corners on the original image are nearly collinear, they likewise don't enclose very many pixels. These few pixels must provide enough information for an entire half of the new image. As a result, the new image will not be very precise.

Now that we know exactly where each point on the new image corresponds to on the old image, we still face the problem that the exact point on the old image may not be an integer. One way to resolve this is to simply choose the closest point. This method is called [Nearest-Neighbor Interpolation](http://en.wikipedia.org/wiki/Nearest-neighbor_interpolation) and gives decent results. Another approach is to take a linear combination of the 4 closest pixels. This method, called [Bilinear Interpolation](http://en.wikipedia.org/wiki/Bilinear_interpolation) gives better results which appear less pixelated, but is slower (for every pixel on the new image, you must consider 4 points on the original instead of a single point for Nearest-Neighbor Interpolation).

Specifically, let the nearest pixels be $$$ul$$$ (upper left), $$$ur$$$ (upper right), $$$ll$$$ (lower left), and $$$lr$$$ (lower right). Additionally, let $$$dist\_{left}$$$ be the distance along the x axis from the exact point to $$$ul$$$ or $$$ur$$$, and let $$$dist\_{top}$$$ be the distance along the y axis from $$$ul$$$ or $$$ur$$$. Both of these distances are between 0 and 1. The pixel value for the exact point should become more dependent on a neighbor when it gets closer to that neighbor. In one dimension, when neighbors $$$l$$$ and $$$r$$$, the pixel value should be $$$(1-dist\_{left} * l) + dist\_{left} * right$$$. Extending to 2 dimensions, the exact pixel value should be:

$$
(1-dist\_{left}) * (1-dist\_{top}) * ul + \\\
(1-dist_{left}) * (dist\_{top}) * ll + \\\
(dist\_{left}) * (1 - dist\_{top}) * ur + \\\
(dist\_{left}) * (dist\_{top}) * lr
$$

There are [many other methods](http://en.wikipedia.org/wiki/Image_scaling) of interpolation which consider [more points](http://en.wikipedia.org/wiki/Bicubic_interpolation), [vectororization](http://research.microsoft.com/en-us/um/people/kopf/pixelart/), [pattern detection](http://en.wikipedia.org/wiki/Hqx), and many other techniques.

=========

###Step 4

Now that we have extracted the notes from the image, the next step is to convert handwriting into text. The process is known as [Optical Character Recognition (OCR)](http://en.wikipedia.org/wiki/Optical_character_recognition). Importantly, this will allow us to search through our notes months/years after writing them.

This step has not been implemented yet. I will likely use some of the following tools:

* [Ocrad](http://www.gnu.org/software/ocrad/)
* [Cuneiform]("http://en.wikipedia.org/wiki/CuneiForm_(software)")
* [ExactImage](http://www.exactcode.de/site/open_source/exactimage/hocr2pdf/)
* [Tesseract](https://code.google.com/p/tesseract-ocr/)

===========
Source: [https://github.com/ben-eysenbach/NoteScanner](https://github.com/ben-eysenbach/NoteScanner)


Ben Eysenbach, [eysenbachbe@gmail.com](mailto:eysenbachbe@gmail.com), 2014.
