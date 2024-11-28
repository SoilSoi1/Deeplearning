import numpy as np

# Multi-dimensional array
def multidim_array():

    # Create arrays
    A = np.array([1, 2, 3, 4])
    B = np.array([[1,2], [3,4], [5,6]])

    # Output the dimension number
    print(np.ndim(A))
    print(str(np.ndim(B)) + '|' + str(B.shape))



if __name__ == "__main__":
    multidim_array()