import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("D:\\my work\\Trafic light\\red\\r5.jpg")
#image = cv2.boxFilter(image , -1 , (3 , 3))
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.namedWindow("HSV Space" , cv2.WINDOW_NORMAL)
cv2.imshow("HSV Space" , hsv)

# red , green and yellow color's range in HSV space

lower_yellow = np.array([20, 50, 90])   
upper_yellow = np.array([40, 255, 255])

low_green = np.array([25, 52, 90]) 
high_green = np.array([102, 255, 255])

low_red1  = np.array([0,50,90])     
high_red1 = np.array([5,255,255])
low_red2  = np.array([175,50,90])    
high_red2 = np.array([180,255,255])

# Masks

mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)

mask1 = cv2.inRange(hsv,low_red1,high_red1)
mask2 = cv2.inRange(hsv,low_red2,high_red2)
mask_r = cv2.bitwise_or(mask1, mask2 )

mask_g = cv2.inRange(hsv , low_green , high_green)


res_y  = cv2.bitwise_and(image, image, mask = mask_y)
res_g  = cv2.bitwise_and(image, image, mask = mask_g)
res_r  = cv2.bitwise_and(image, image, mask = mask_r)

cv2.namedWindow("Yellow Mask" , cv2.WINDOW_NORMAL)
cv2.imshow("Yellow Mask" , mask_y)

cv2.namedWindow("Green Mask" , cv2.WINDOW_NORMAL)
cv2.imshow("Green Mask" , mask_g)

cv2.namedWindow("Red Mask" , cv2.WINDOW_NORMAL)
cv2.imshow("Red Mask" , mask_r)

cv2.namedWindow("Red area" , cv2.WINDOW_NORMAL)
cv2.imshow("Red area" , res_r)

cv2.namedWindow("yellow area" , cv2.WINDOW_NORMAL)
cv2.imshow("yellow area" , res_y)

cv2.namedWindow("Green area" , cv2.WINDOW_NORMAL)
cv2.imshow("Green area" , res_g)

r = 0
g = 0
y = 0

for i in range(mask_y.shape[0]):
	for j in range(mask_y.shape[1]):
		if mask_y[i , j] == 255 :
			y = y + 1
for i in range(mask_g.shape[0]):
	for j in range(mask_g.shape[1]):
		if mask_g[i , j] == 255 :
			g = g + 1

for i in range(mask_r.shape[0]):
	for j in range(mask_r.shape[1]):
		if mask_r[i , j] == 255 :
			r = r + 1

n = [[y] , [g] , [r]]

plt.subplot(121)
plt.imshow(n , cmap = 'cool')
plt.xlim([-0.5 , 0.5])
plt.xticks([0] , ['number of pixels'])
plt.yticks([0 , 1 , 2] , ['yellow' , 'green' , 'red'])
plt.text(0,0,n[0][0])
plt.text(0,1,n[1][0])
plt.text(0,2,n[2][0])

plt.subplot(122)

mx = y
c = 0
if (r > mx) :
        mx = r
        c = 1
if (g > mx) :
        mx = g
        c = 2

col = ['yellow' , 'red' , 'green']
board = np.ones([25 , 25])
plt.imshow(board , cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.text(10 , 12.5 , col[c] , fontfamily = 'cursive' , fontsize = 'xx-large' , fontstyle = 'oblique' , fontweight = 'black' , color = col[c])
plt.title("     ****  Predicted color  *****   ")
plt.show()

