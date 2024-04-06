import numpy as np

class Convolution_Transform:

    border_calculation="ignore"
    constant_padding_value=0
    convolved_matrix=None
    slice_x_addition=0
    slice_y_addition=0
    kernel_center=[0,0]

    def __init__(self,**keyword):
        self.image=keyword["image"]
        self.kernel=keyword["kernel"]
        
        if("kernel_center" in keyword):
            self.kernel_center=keyword['kernel_center']
        else :
            i=j=int(self.kernel.shape[0]/2)
            self.kernel_center=[i,j]
            
        if("padding" in keyword):
            self.border_calculation=keyword['padding']
            
        if("constant_value" in keyword):
            self.constant_padding_value=keyword['constant_value']


    def __str__(self) -> str:
        return("convolutional transform object")
    
    def constant_padding(self):
        image_width=self.image.shape[0]
        image_height=self.image.shape[1]

        kernel_width=self.kernel.shape[0]
        kernel_height=self.kernel.shape[1]

        new_image=np.full((image_width+(kernel_width*2),image_height+(kernel_height*2)),self.constant_padding_value)


        for i in range(0,image_width):

            for j in range(0,image_height):

                new_image[i+kernel_width,j+kernel_height]=self.image[i,j]

        self.image=new_image
    
    def padding(self):
        if(self.border_calculation=="ignore"):
            self.constant_padding()
            
        if(self.border_calculation=="constant"):
            self.constant_padding()
            
        if(self.border_calculation=="reverse"):
            self.ignore_padding()

        #---add additional value for next process
        self.slice_x_addition=self.kernel.shape[0]
        self.slice_y_addition=self.kernel.shape[1]

    
    def slicer(self,target_pixel):
        n=self.kernel.shape[0]
        kernel_i=self.kernel_center[0]
        kernel_j=self.kernel_center[1]
        slice=np.zeros((n,n))

        image_i=target_pixel[0]+self.slice_x_addition
        image_j=target_pixel[1]+self.slice_y_addition
        
        #---reset dimension of image center and transformer center---
        image_i=image_i-kernel_i
        image_j=image_j-kernel_j
        kernel_i=0
        kernel_j=0

        slice_i=0
        for i in range(image_i,image_i+n):

            slice_j=0
            for j in range(image_j,image_j+n):
                slice[slice_i,slice_j]=self.image[i,j]
                slice_j=slice_j+1

            slice_i=slice_i+1
        
        return slice
    
    def convolver(self,sliced_image):
        result=0
        for i in range(0,sliced_image.shape[0]):
            for j in range(0,sliced_image.shape[1]):
                convolved=sliced_image[i,j]*self.kernel[i,j]
                result=result+convolved
        return result
    
    def apply(self):
        image_width=self.image.shape[0]
        image_height=self.image.shape[1]

        self.kernel=np.flip(self.kernel)

        self.convolved_matrix=np.zeros((image_width,image_height))

        self.constant_padding()

        for image_i in range(0,image_width):
            for image_j in range(0,image_height):
                sliced_image=self.slicer([image_i+1,image_j+1])
                convolved=self.convolver(sliced_image)
                if(convolved > 255):
                    convolved=255
                if(convolved < 0):
                    convolved=0
                self.convolved_matrix[image_i,image_j]=convolved