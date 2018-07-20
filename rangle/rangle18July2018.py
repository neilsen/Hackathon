import sys
import cv2
import numpy as np
import math

font = cv2.FONT_HERSHEY_SIMPLEX
#img = cv2.imread("./KHARKOF.jpg") # works for both
##img = cv2.imread("./KARL.jpg")   # works for both, lines drawn too high
#img = cv2.imread("./ABOVE.jpg")
#img = cv2.imread("./oneLine.jpg")
img = cv2.imread("./CUSTER.jpg")

img2 = img.copy()

#
# Color thresholding based on Red color in BGR image
#
# Just 160-255 in red --> white, all other pixels black
#
BGR_MIN = np.array([0, 0, 160],np.uint8)
BGR_MAX = np.array([255, 255, 255],np.uint8)
imgThresh = cv2.inRange(img, BGR_MIN, BGR_MAX)
cv2.imwrite('imgThresh.jpg', imgThresh)

#
# Binary threshold and compute sure foreground
#
kernel = np.ones((1,1),np.uint8)
opening = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)
blur = cv2.GaussianBlur(opening,(1,1),0)
ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=1)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)

ret, sure_fg = cv2.threshold(dist_transform,0.8*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
cv2.imwrite('sure_fg.jpg',sure_fg)

#
# Connected components in sure foreground
#
# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(sure_fg, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
print 'num_labels = ' + str(num_labels)
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]
print 'centroids: '
for i in range(1,num_labels):
    print str(output[2][i][0]) + ", " + str(output[2][i][1]) + ": " + str(output[2][i][2]) + ", " + str(output[2][i][3])
    cv2.rectangle(img2, (output[2][i][0], output[2][i][1]), (
                    output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)
    print 'centroid: ' + str(centroids[i][0]) + ", " + str(centroids[i][1])
    
    cv2.imwrite('connectedComponents_'+str(i)+'.jpg', img2)

    scale_factor_x = 10
    scale_factor_y = 20
    a1 = int(centroids[i][1]-scale_factor_x*output[2][1][3])
    a2 = int(centroids[i][1]+scale_factor_x*output[2][1][3])
    b1 = int(centroids[i][0]-scale_factor_y*output[2][1][2])
    b2 = int(centroids[i][0]+scale_factor_y*output[2][1][2])

    if (a1<0):
        a1=0
    if (a2>imgThresh.shape[0]):
        a2=imgThresh.shape[0]
    if (b1<0):
        b1=0
    if (b2>imgThresh.shape[1]):
        b2=imgThresh.shape[1]
        
    imgSmall = imgThresh [ a1:a2,b1:b2 ]
    img2Small = img2 [ a1:a2,b1:b2 ]
    xoff = b1
#    yoff = int(centroids[i][1]-scale_factor_y*output[2][1][3]) 
    yoff = a1
    
    rows = imgSmall.shape[0]
    cols = imgSmall.shape[1]
    #imgSmall[0:int(rows*0.6),0:cols-1]=0
    chop_ratio = 0.5
    
    imgSmall = imgSmall[int(rows*chop_ratio):rows-1,0:cols-1]
    img2Small = img2Small[int(rows*chop_ratio):rows-1,0:cols-1]
#    print 'yoff:' + str(yoff) + ', rows*0.62:' + str(int(rows*0.62))
    yoff = yoff + int(rows*(1.0-chop_ratio))
    cv2.imwrite('imgSmall' + str(i) + '.jpg', imgSmall)

#    kernel = np.ones((5,5),np.uint8)
#    kernel = np.ones((3,3),np.uint8)
#    imgSmall = cv2.erode(imgSmall,kernel,iterations = 1)

    cv2.imwrite('imgSmallEroded'+str(i)+'.jpg', imgSmall)

    # Call our own version of hough_line
    #accumulator, thetas, rhos = hough_line(imgSmall)
    #
    # Easiest peak finding based on max votes
    #idx = np.argmax(accumulator)
    #rho = rhos[idx / accumulator.shape[1]]
    #theta = thetas[idx % accumulator.shape[1]]
    #print "rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta))

    # This returns an array of r and theta values
    #lines = cv2.HoughLines(gray,1,np.pi/180, 200)

    height, width, channels = img2Small.shape
    xc = width/2.0
    yc = 0
    
    minLineLength = 5
    maxLineGap = 100

    img2S = img2Small.copy()
    img3 = img2.copy()
    max_degree = -90
    min_degree = 90

    lineNo = 0
    lines2 = cv2.HoughLines(imgSmall,1,np.pi/180, 200) # was 90, 120
    if (lines2 is not None):
        print "Number of lines: " + str(lines2.size)
        for line in lines2:
            for r,theta in line:
                print "r: " + str(r) + ", theta: " + str(theta) + ", deg: " + str(math.degrees(theta))
                degrees = math.degrees(theta)
                # Stores the value of cos(theta) in a
                a = np.cos(theta)
                # Stores the value of sin(theta) in b
                b = np.sin(theta)
     
                # x0 stores the value rcos(theta)
                x0 = a*r
     
                # y0 stores the value rsin(theta)
                y0 = b*r
     
                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                x1 = int(x0 + 1200*(-b))
     
                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                y1 = int(y0 + 1200*(a))
 
                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                x2 = int(x0 - 1200*(-b))
     
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                y2 = int(y0 - 1200*(a))
                
                distMax = 25
                
                d = math.fabs((y2-y1)*xc-(x2-x1)*yc+x2*y1-y2*x1)/math.sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))
                print 'd: ' + str(d)

                if (r < 0.0):
                    degrees = degrees - 180
                if ((degrees > max_degree) and (d < distMax)):
                    max_degree = degrees
                if ((degrees < min_degree) and (d < distMax)):
                    min_degree = degrees

                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                # (0,0,255) denotes the colour of the line to be 
                #drawn. In this case, it is red.
#                if (lineNo==13):
                img2Sline = img2Small.copy()
                cv2.line(img2Sline,(x1,y1), (x2,y2), (0,0,255),2)
                cv2.line(img2S,(x1,y1), (x2,y2), (0,0,255),2)
                td = '< : ' +str(int(math.degrees(theta)))
                tx = 'x : ' + str(r*math.cos(theta))
                ty = 'y : ' + str(r*math.sin(theta))
                cv2.circle(img2Sline,(int(xc),int(yc+3)), 6, (255,0,128), -1)
                cv2.putText(img2Sline, td, (10,220), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img2Sline, 'd : ' + str(d), (10,260), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img2Sline, tx, (10,300), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img2Sline, ty, (10,340), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imwrite(str(i)+'line'+str(lineNo)+'.jpg', img2Sline)
                lineNo=lineNo+1
        cv2.imwrite('all_linesDetected'+str(i)+'.jpg', img2S)

    print 'Max degrees: ' + str(max_degree) + ', Min degrees: ' + str(min_degree)
    if (lines2 is not None):
        for line in lines2:
            for r,theta in line:
                degrees = math.degrees(theta)
                if (r < 0.0):
                    degrees = degrees - 180
                if ((degrees == max_degree)or(degrees == min_degree)):
                    # Stores the value of cos(theta) in a
                    a = np.cos(theta)
                    # Stores the value of sin(theta) in b
                    b = np.sin(theta)
     
                    # x0 stores the value rcos(theta)
                    x0 = a*r
     
                    # y0 stores the value rsin(theta)
                    y0 = b*r
     
                    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
                    x1 = int(x0 + 1200*(-b))
     
                    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
                    y1 = int(y0 + 1200*(a))
 
                    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
                    x2 = int(x0 - 1200*(-b))
     
                    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
                    y2 = int(y0 - 1200*(a))
     
                    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
                    # (0,0,255) denotes the colour of the line to be 
                    # drawn. In this case, it is red. 
                    if (degrees == max_degree):
                        cv2.line(img2S,(x1,y1), (x2,y2), (255,0,0),3)
                        cv2.line(img3,(xoff+x1,yoff+y1),(xoff+x2,yoff+y2),(255,0,0),3)
                    else:
                        cv2.line(img2S,(x1,y1), (x2,y2), (0,255,0),3)
                        cv2.line(img3,(xoff+x1,yoff+y1),(xoff+x2,yoff+y2),(0,255,0),3)

     
        # All the changes made in the input image are written to linesDetected.jpg
        cv2.imwrite('linesDetected'+str(i)+'.jpg', img2S)

    if (max_degree > min_degree):
        print 'For case: ' + str(i) + ', max angle: ' + str(max_degree - min_degree) + ' = (' + str(max_degree) + ' - ' + str(min_degree) + ')'
        cv2.putText(img3, str(max_degree - min_degree), (10,80), font, 3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imwrite('biglinesDetected'+str(i)+'.jpg', img3)

def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=128):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - Boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos
