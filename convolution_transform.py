import numpy as np

class Convolution_Transform:

    border_calculation="ignore"
    constant_padding_value=0
    convolved_matrix=None
    slice_x_addition=0
    slice_y_addition=0

    def __init__(self,**keyword):
        self.image=keyword["image"]
        self.transform=keyword["transform"]
        
        if("padding" in keyword):
            self.border_calculation=keyword['padding']
            
        if("constant_value" in keyword):
            self.constant_padding_value=keyword['constant_value']


    def __str__(self) -> str:
        return("convolutional transformer object")
    
    def constant_padding(self):
        image_width=self.image.shape[0]
        image_height=self.image.shape[1]

        transform_width=self.transform["matrix"].shape[0]
        transform_height=self.transform["matrix"].shape[1]

        new_image=np.full((image_width+(transform_width*2),image_height+(transform_height*2)),self.constant_padding_value)


        for i in range(0,image_width):

            for j in range(0,image_height):

                new_image[i+transform_width,j+transform_height]=self.image[i,j]

        self.image=new_image
    
    def padding(self):
        if(self.border_calculation=="ignore"):
            self.constant_padding()
            
        if(self.border_calculation=="constant"):
            self.constant_padding()
            
        if(self.border_calculation=="reverse"):
            self.ignore_padding()

        #---add additional value for next process
        self.slice_x_addition=self.transform["matrix"].shape[0]
        self.slice_y_addition=self.transform["matrix"].shape[1]

    
    
    def slicer(self,target_pixel):
        n=self.transform["matrix"].shape[0]
        transform_i=self.transform["center"][0]
        transform_j=self.transform["center"][1]
        slice=np.zeros((n,n))

        image_i=target_pixel[0]+self.slice_x_addition
        image_j=target_pixel[1]+self.slice_y_addition
        
        #---reset dimension of image center and transformer center---
        image_i=image_i-transform_i
        image_j=image_j-transform_j
        transform_i=0
        transform_j=0

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
                convolved=sliced_image[i,j]*self.transform["matrix"][i,j]
                result=result+convolved
        return result
    
    def apply(self):
        image_width=self.image.shape[0]
        image_height=self.image.shape[1]
        image_width=self.image.shape[0]
        image_height=self.image.shape[1]

        self.transform["matrix"]=np.flip(self.transform["matrix"])

        self.convolved_matrix=np.zeros((image_width,image_height))

        self.constant_padding()

        for image_i in range(0,image_width):
            for image_j in range(0,image_height):
                sliced_image=self.slicer([image_i,image_j])
                convolved=self.convolver(sliced_image)
                print(convolved)
                self.convolved_matrix[image_i,image_j]=convolved


        


# image=np.array([
#     [1,2,3,4],
#     [5,6,7,8],
#     [9,10,11,12,],
#     [13,14,15,16],
# ])

image=np.array([
    [1,2,3],
    [5,6,7],
    [9,10,11],
])

# matrix=np.array([
#     [0, 0, 0],
#     [0, 1, 0],
#     [0, 0, 0],
#     ])

matrix=np.array([
    [1,0],
    [1,1],
    ])

transform={
    "matrix":matrix,
    "center":[0,0]
}

test=Convolution_Transform(image=image,transform=transform)

test.apply()
print(test.convolved_matrix)