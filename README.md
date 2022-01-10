# Google Face Expression Comparison Dataset Creation Library

A lot of effort in solving any machine learning problem goes into
preparing the data. This SDK is intended to get the image dataset from **Google Face Expression Comparison Dataset** in the format required for solving machine learning tasks.
>The Google Face Expression Dataset is in CSV file format. It contains both test and train CSV files. Both the train and the test CSV files have identical fields. Each line in the CSV file corresponds to one data sample and contains three images specified by their urls along with their face bounding boxes in normalised form. 
>
 
## Usage

As an example, let us assume:
* **input_folder** is the folder in which the CSV file of  is stored.
* **images_folder** is the folder which saves full images.
* **faces_folder** is the folder which saves only the face images.
    > We use the bounding box information to slice images faces.
    
    **Get images from CSV file**:
    ```
    dg = GoogleFECdatasetCreator(input_folder)
    valid_indexes = dg.create_images_repository(images_folder, start_index, end_index)
  ```
  **Get face images from CSV file**
   ```
  > dg = GoogleFECdatasetCreator(input_folder)
  > valid_indexes = dg.create_sliced_images(faces_folder, start_index, end_index)
  ```
  
  **Get face images directly from all_output_folder**
  ```
  > dg = GoogleFECdatasetCreator(input_folder)
  > valid_indexes = dg.create_sliced_images_from_folder(images_folder,
  > faces_folder)
  ```
  
 -- **valid_indexes** returned above represents the list of indexes where all the three images could be downloaded from their respective urls. Each index corresponds to the index of a row in the CSV file containing three image urls as per the following conventions: 
   >  Index specified as 0 corresponds to the index of the first row in csv file
  
   The indexes not listed represent those row indexes where the 
    **image was not found** and hence cannot be used for training or testing purposes.
    
* The image names in the output file are stored as "**index_0.jpg**" or "**index_1.jpg**" or "**index_2.jpg**"
    


