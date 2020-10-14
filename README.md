# Thumb-Gestures-Detection
Detecting gestures of thumb using CNN and OpenCV.

![](outputs/thumb.gif)

For better visuals you watch [this](https://www.youtube.com/watch?v=6LE1yf1mA1w) or download [this](https://github.com/greatsharma/Thumb-Gestures-Detection/blob/master/outputs/thumb.mp4) and see locally.

I used `background subtraction` for extracting the hand pixels and then passed to a CNN. I trained the CNN using the same approach i.e., on extracted pixels of hands. Here is a sample of data on which I train the model.

![](outputs/up_img.png)

I custom created the entire dataset using opencv, you can download the entire dataset from [here](https://github.com/greatsharma/Thumb-Gestures-Detection/blob/master/data.zip).

The biggest challenge was to extract the hand as I am not using any object-detection API so it was even more challenging for me. I used two background subtraction techniques 1.`AbsDiff` 2.`MOG`. AbsDiff gave excellent result whereas MOG does bad in these conditions. I uploaded both of them, you can test both by your own.

### How to use
Simply run `python test.py`. It takes only one cli argument `mode` which is either `debug` or `prod` default is `prod`. As I am using background subtraction so **once you run the program you need to wait for 4 seconds** so that the blue window capture what's inside it and after that you can place you hand inside it to show the gestures. Also **after 4 seconds you cannot change the location of your window** because in that case background subtraction won't work beacuse it assumes the background to be constant or nearly constant.  

I gave 4 second time so that user get enough time to get ready otherwise 1 second is also enough.
