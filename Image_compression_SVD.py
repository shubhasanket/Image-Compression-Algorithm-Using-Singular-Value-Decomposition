'''
Image compression using singular value decomposition (SVD)
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def calc_SVD(matrix):
    '''
    This routine calculates the SVD
    '''
    A_T = matrix.transpose()
    AA_T = matrix.dot(A_T) # calculating A*A.T
    print("AA_T =", AA_T) 
##    print(AA_T.shape)
    sigma_squared, U = np.linalg.eig(AA_T) # U is already in the required shape, no transpose is required
    print("sigma squared = ",sigma_squared)
    sigma = np.sqrt(sigma_squared) # sigma is contains the singular values on its diagonal
    print("U: ",U)
    V_T = []
    for i in range (matrix.shape[0]): # calculating the matrix V transpose
        V_T.append((A_T.dot(U[:,i])*(1/sigma[i]))) # formula for calculating the columns of V
        
    V_T = np.array(V_T)
    print("V transpose =", V_T)

    return U, sigma, V_T


def compute_outer_products_till_R(U,sigma,V_T,A,r=50):
    '''
    This routine calculates the approximation for a given rank
    '''
    compressed_matrix = np.zeros(A.shape)
    for i in range (r):
        compressed_matrix += np.outer(U[:,i],V_T[i])*sigma[i]

    return compressed_matrix
    

def clip(h,w,l):
    '''
    This routines clips values above 255 and below 0.
    The values cross the designated range due to errors incurred during calculations
    '''
    processed_matrix = []
    for i in range (h):
        processed_matrix.append([])
        for j in range (w):
            if round(l[i][j]) < 0:
                processed_matrix[i].append(0)
            elif round(l[i][j]) > 255:
                processed_matrix[i].append(255)
            else:
                processed_matrix[i].append(round(l[i][j]))
    return np.array(processed_matrix)


def display(h,w,l):
    '''
    This routine displays the processed image
    '''
    s = clip(h, w, l)
    img = Image.fromarray(l)
    img.show()

def calc_frobenius_norm(A,l):
    '''
    This routine calculates the Frobenius norm.
    The lower the Frobenius norm, the closer is the approximation to
    the original image
    '''
    F = np.subtract(A,l)
    for i in range (len(F)):
        for j in range (len(F[0])):
            F[i][j] = (F[i][j])**2
    
    s = 0
    for u in F:
        s += sum(u)

    return sqrt(s)

def plot_frobenius(A,u,sig,v,stepsize):
    '''
    This routine plots rank vs Frobenius norm
    '''
    start = 5
    x = [x for x in range (start, A.shape[0]-start, stepsize)]
##    print(x)
    # y will contain the frobenius norm of all ranks in the list x
    # This might take a while, 4 to 5 mins
    y = []
    compressed_matrix = np.zeros(A.shape)
    for i in range (A.shape[0]-start):
        compressed_matrix += np.outer(u[:,i],v[i])*sig[i]
            
        if ((i+1)-start)%stepsize == 0:
            y.append(calc_frobenius_norm(A, compressed_matrix))
##    print(y)
    plt.plot(x,y)
    plt.xlabel("Rank")
    plt.ylabel("Frobenius Norm")
    plt.title("Frobenius norms with step size " + str(stepsize))
    plt.show()
    
def main():
    # Opening image
    im = Image.open(r'C:\Users\Shubham\AppData\Local\Programs\Python\Python39\Workspace\Linear_Algebra\3\meditation.tif') 
##    im = Image.open(r'C:\Users\Shubham\AppData\Local\Programs\Python\Python39\Workspace\Linear_Algebra\3\Illusion.tif')
##    im = Image.open(r'C:\Users\Shubham\AppData\Local\Programs\Python\Python39\Workspace\Linear_Algebra\3\Pensive.tif')
##    im = Image.open(r'C:\Users\Shubham\AppData\Local\Programs\Python\Python39\Workspace\Linear_Algebra\3\Sailors.tif')
    ##im.show()

    # Creating a 3D matrix from the .tif image
    im_array = np.array(im) 

    # Dimensions of the matrix
    height, width, no_bytes = im_array.shape
    
    # Creating a 2D matrix from im_array. As the picture is monochromatic, the values for RGB are the same
    l_2D = [] 
    for i in range (height):
        l_2D.append([])
        for j in range (width):
            l_2D[i].append(im_array[i][j][0])
    l_2D = np.array(l_2D, np.float64) # have to include float64 or else getting negative eig values

    # Computing the singular value decomposition of the 2D matrix where u*sigma*v = 2D matrix in theory
    u, sig, v = calc_SVD(l_2D)
##    
##    # Trying approximations for different values of the rank
##    rep = [x for x in range (0, 100, 5)]
##    for i in rep:
##        l = compute_outer_products_till_R(u,sig,v,l_2D,r=i)
##        print("Displaying image for rank:", i)
##        display(height, width, l)
##        s = input("Press Enter to continue or 'S' to stop ")
##        if s == "S":
##            break
    
    # This routines prompts an input for the rank for which an approximation will be generated,
    # higher the rank, better the approximation, bigger the file size (theoretically)
    end_display = False
    while not end_display:
        i = eval(input("Enter the rank(max value is "+str(height if height < width else width)+"): "))
        l = compute_outer_products_till_R(u,sig,v,l_2D,r=i) # Approximation routine
        print("Displaying image for rank", i)
        display(height, width, l) # Display the compressed image
        print("The Frobenius Norm is ", calc_frobenius_norm(l_2D,l))
        s = input("To repeat press 'Enter', to continue enter 'n'")
        if s == 'n':
            end_display = True
##    print("Done")
    # This variable is used to plot the Frobenius norm of the compressed matrix
    step_size = 10 
    # plotting the Frobenius norm for different ranks
    plot_frobenius(l_2D,u,sig,v,step_size)

    # Routine to display a satisfactory compressed image, whose rank will decided from the displayed plot  
    i = eval(input("Enter the rank: "))
    l = compute_outer_products_till_R(u,sig,v,l_2D,r=i)
    print("Displaying image for rank:", i)
    display(height, width, l)
    print("The Frobenius Norms is: ", calc_frobenius_norm(l_2D,l))
    print("End")
main()
