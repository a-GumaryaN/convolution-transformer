import numpy as np

def padding(**keywords):
    
    pad_value=0

    matrix=keywords["matrix"]
    pad_size=keywords["pad_size"]
    pad_value=keywords["pad_value"]

    padded_matrix=np.full((matrix.shape[0]+(pad_size*2),matrix.shape[1]+(pad_size*2)) , pad_value)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            padded_matrix[i+pad_size,j+pad_size]=matrix[i,j]
    return padded_matrix


def convolver(**keywords):

    matrix=keywords["matrix"]
    kernel=keywords["kernel"]
    center=keywords["center"]
    target=keywords["target"]

    k_i=center[0]
    k_j=center[1]
    m_i=target[0]
    m_j=target[1]
    min_i=m_i - k_i
    max_i=min_i + kernel.shape[0]
    
    min_j=m_j - k_j
    max_j=min_j + kernel.shape[1]

    result=0

    k_i=0
    k_j=0

    for i in range(min_i,max_i):
        k_j=0
        for j in range(min_j,max_j):

            result=result + matrix[i,j] * kernel[k_i,k_j]
            k_j=k_j+1

        k_i=k_i+1

    return result

def convolution_transform(**keywords):
    matrix=keywords["matrix"]
    kernel=keywords["kernel"]
    parameter=keywords["parameter"]
    pad_value=keywords["pad_value"]
    result=[]
    for i in range(pad_value,matrix.shape[0]-pad_value):
        row=[]
        for j in range(pad_value,matrix.shape[1]-pad_value):
            convolved=convolver(matrix=matrix,kernel=kernel,center=[1,1],target=[i,j])
            convolved=convolved*parameter
            if convolved > 255 :
                convolved=255
            if convolved < 0:
                convolved=0
            row.append(convolved)
        result.append(row)
    return np.array(result)