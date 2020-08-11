import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "checker_board.png"
path2 = "house.jpg"
img = cv2.imread(path)
img2 = cv2.imread(path2)


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    kH,kW = conv_filter.shape
    (imH,imW) = img.shape
    result = np.zeros(img.shape)
    pad = int((kH-1)/2)
    
    for y in range(imH-kH):
        for x in range(imW-kW):
            window = img[y:y+kH,x:x+kW]
            result[y+pad,x+pad] = (conv_filter * window).sum()

    return result


# #https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def gauss_filter(shape = (3,3), sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def harris(img):

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    g = gauss_filter((5,5),1)

    dx = np.array([[-1,0,1],
                   [-1,0,1],
                   [-1,0,1]])
    dy = dx.transpose()

    Ix = conv2(bw,dx)
    Iy = conv2(bw,dy)

    Ix2 = conv2(np.power(Ix,2),g)
    Iy2 = conv2(np.power(Iy,2),g)
    Ixy = conv2((Ix*Iy),g)
    
    det = (Ix2 * Iy2) - (np.power(Ixy,2))

    trace = Ix2 + Iy2
    k = 0.04
    r = det - k*(np.power(trace,2))

    #Non-Max Suppression
    maxima = r.max()
    corner_points = np.array([])
    detected_img = img.copy()
    thresh = 0.01

    window_size = 3
    gap = window_size - 1 // 2
    row,col = r.shape

    for h in range(gap, row-(gap + 1)):
        for w in range(gap, col-(gap + 1)):
                #Define the 2D space to focus on (window)
            window = r[h-gap:h+(gap+1),w-gap:w+(gap+1)]
                #Condition to meet : Value must be the largest within the 2D window and is larger than the product of maxima and threshold
                #This is to prevent multiple detections around the same corner
            if r[h,w] > maxima * thresh and r[h,w] == np.max(window):
                    #Creating a red box around the detected corner
                detected_img[h-1:h+1,w-1:w+1] = [255,0,0]
                if(corner_points.size == 0):
                    corner_points = np.array([h,w])
                else:
                    corner_points = np.vstack((corner_points,[h,w]))
    return corner_points, detected_img

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
filename = 'checker_board.png'
filename2 = "house.jpg"
ocv = cv2.imread(filename)
ocv2 = cv2.imread(filename2)

gray = cv2.cvtColor(ocv, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(ocv2, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
gray2 = np.float32(gray2)

dst = cv2.cornerHarris(gray,2,3,0.04)
dst2 = cv2.cornerHarris(gray2,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
dst2 = cv2.dilate(dst2, None)

# Threshold for an optimal value, it may vary depending on the image.
ocv[dst>0.01*dst.max()]=[0,0,255]
ocv2[dst2>0.01*dst2.max()]=[0,0,255]




corner, detected = harris(img)
corner2, detected2 = harris(img2)


plt.subplot(221)
plt.title("Checker Board")
plt.imshow(detected)

plt.subplot(222)
plt.title("Image of a House")
plt.imshow(detected2)

#For comparison
plt.subplot(223)
plt.title("OpenCV Harris Corner Detector")
plt.imshow(ocv)

plt.subplot(224)
plt.title("OpenCV Harris Corner Detector")
plt.imshow(ocv2)

plt.show()