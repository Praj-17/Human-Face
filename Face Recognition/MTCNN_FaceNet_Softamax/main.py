from DataBuilder.data_builder import Data_builder
from DataBuilder.augmentation import Augmentation
import os


person = "Sakshi"
print(person)
data_builder = Data_builder(person)
data_builder.get_images()

# Augmentation(f"{data_builder.path}\\",f"{data_builder.path}\\","aug",15)
    
   

