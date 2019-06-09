import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import tempfile
from skimage.feature import match_template
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.io import imread
from skimage import restoration
from skimage import measure
from sklearn.externals import joblib
from django.template.loader import render_to_string
from django.conf import settings
    
# characters that should be clearly examined using template matching
confusing_chars = {'2', 'Z', 'B', '8', 'D', '0', '5', 'S', 'Q', 'R', '7'}
# a dictionary that keeps track of characters that are similar to the
# confusing characters
similar_characters = {
    '2':['Z'], 'Z':['2', '7'], '8':['B'], 'B':['8', 'R'], '5':['S'], 'S':['5'],
    '0':['D', 'Q'], 'D':['0', 'Q'], 'Q':['D', '0'], '7':['Z']
}

def license_plate_extract(request):
    f = request.FILES['image[]']
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        for chunk in f.chunks():
            tmpfile.write(chunk)

    start_time = time.time()
    models_folder = settings.ML_MODELS_ROOT
    pre_process = PreProcess(tmpfile.name)
    
    plate_like_objects = pre_process.get_plate_like_objects()
    plotting = Plotting()
    plotting.plot_cca(pre_process.full_car_image, pre_process.plate_objects_cordinates)
    number_of_candidates = len(plate_like_objects)

    if number_of_candidates == 0:
        license_plate = []
    elif number_of_candidates == 1:
        license_plate = pre_process.inverted_threshold(plate_like_objects[0])
    else:
        license_plate = pre_process.validate_plate(plate_like_objects)

    data = dict()

    if len(license_plate) == 0:
        end_time = str(time.time() - start_time)  + ' sec'
        data['end_time'] = end_time
        data['error_message'] = "License plate could not be located"
        return data
    else:
        ocr_instance = OCROnObjects(license_plate)

        if ocr_instance.candidates == {}:
            end_time = str(time.time() - start_time)  + ' sec'
            data['end_time'] = end_time
            data['error_message'] = "No character was segmented"
            return data
        else:
            plotting.plot_cca(license_plate, ocr_instance.candidates['coordinates'])
            deep_learn = DeepMachineLearning()
            text_result = deep_learn.learn(ocr_instance.candidates['fullscale'],
                os.path.join(models_folder, 'GaussianNB', 'GaussianNB.pkl'),(20, 20))

            text_phase = TextClassification()
            scattered_plate_text = text_phase.get_text(text_result)
            plate_text = text_phase.text_reconstruction(scattered_plate_text,
                ocr_instance.candidates['columnsVal'])
            
            end_time = str(time.time() - start_time)  + ' sec'
            data['end_time'] = end_time
            data['plate_text'] = plate_text
            return data

class DeepMachineLearning():
    
    def __init__(self):
        self.letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

    def learn(self, objects_to_classify, modelDir, tuple_size):
        model = self.load_model(modelDir)
        return self.classify_objects(objects_to_classify, model, tuple_size)
        
    def classify_objects(self, objects, model, tuple_resize):
        """
        uses the predict method in the model to predict the category(character)
        that the image belongs to

        Parameters
        ___________
        objects: Numpy array
        """
        classificationResult = []
        for eachObject in objects:
            eachObject = resize(eachObject, tuple_resize)
            eachObject = eachObject.reshape(1, -1)
            result = model.predict(eachObject)
            probabilities = model.predict_proba(eachObject)
            result_index = self.letters.index(result[0])
            prediction_probability = probabilities[0, result_index]
            # template matching when necessary
            if result[0] in confusing_chars and prediction_probability < 0.15:
                templatematching = TemplateMatching()
                result[0] = templatematching.template_match(result[0],
                    eachObject, os.path.join(os.path.dirname(os.path.realpath(
                        __file__)), 'training_data', 'train20X20'))
            classificationResult.append(result)
        
        return classificationResult
        
    def load_model(self, model_dir):
        """
        loads the machine learning using joblib package
        model_dir is the directory for the model
        loading of the model has nothing to do with the classifier used
        """
        model = joblib.load(model_dir)
        return model

class OCROnObjects():
    
    def __init__(self, license_plate):
        character_objects = self.identify_boundary_objects(license_plate)
        self.get_regions(character_objects, license_plate)
        
    def identify_boundary_objects(self, a_license_plate):
        labelImage = measure.label(a_license_plate)
        character_dimensions = (0.4*a_license_plate.shape[0], 0.85*a_license_plate.shape[0], 0.04*a_license_plate.shape[1], 0.15*a_license_plate.shape[1])
        minHeight, maxHeight, minWidth, maxWidth = character_dimensions
        regionLists = regionprops(labelImage)
        return regionLists
    
    def get_regions(self, character_objects, a_license_plate):
        """
        used to map out regions where the license plate charcters are 
        the principle of connected component analysis and labelling
        were used

        Parameters:
        -----------
        a_license_plate: 2D numpy binary image of the license plate

        Returns:
        --------
        a dictionary containing the index
        fullscale: 3D array containig 2D array of each character 
        columnsVal: 1D array the starting column of each character
        coordinates:
        """
        cord = []
        counter=0
        column_list = []
        character_dimensions = (0.35*a_license_plate.shape[0], 0.60*a_license_plate.shape[0], 0.05*a_license_plate.shape[1], 0.15*a_license_plate.shape[1])
        minHeight, maxHeight, minWidth, maxWidth = character_dimensions
        for regions in character_objects:
            minimumRow, minimumCol, maximumRow, maximumCol = regions.bbox
            character_height = maximumRow - minimumRow
            character_width = maximumCol - minimumCol
            roi = a_license_plate[minimumRow:maximumRow, minimumCol:maximumCol]
            if character_height > minHeight and character_height < maxHeight and character_width > minWidth and character_width < maxWidth:
                if counter == 0:
                    samples = resize(roi, (20,20))
                    cord.append(regions.bbox)
                    counter += 1
                elif counter == 1:
                    roismall = resize(roi, (20,20))
                    samples = np.concatenate((samples[None,:,:], roismall[None,:,:]), axis=0)
                    cord.append(regions.bbox)
                    counter+=1
                else:
                    roismall = resize(roi, (20,20))
                    samples = np.concatenate((samples[:,:,:], roismall[None,:,:]), axis=0)
                    cord.append(regions.bbox)
                column_list.append(minimumCol)
        if len(column_list) == 0:
            self.candidates = {}
        else:
            self.candidates = {
                        'fullscale': samples,
                        'coordinates': np.array(cord),
                        'columnsVal': column_list
                        }
        
        return self.candidates

class Plotting:

    def plot_cca(self, image, objects_cordinates):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.imshow(image, cmap=plt.get_cmap('gray'))

        for each_cordinate in objects_cordinates:
            min_row, min_col, max_row, max_col = each_cordinate
            bound_box = mpatches.Rectangle((min_col, min_row), max_col - min_col,
                max_row - min_row, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(bound_box)

        #plt.show()

class PreProcess():
    
    def __init__(self, image_location):
        """
        reads the image in grayscale and thresholds the image

        Parameters:
        -----------

        image_location: str; full image directory path
        """
        self.full_car_image = imread(image_location, as_gray=True)
        
        self.full_car_image = self.resize_if_necessary(self.full_car_image)

        self.binary_image = self.threshold(self.full_car_image)
        
    def denoise(self, imgDetails):
        return restoration.denoise_tv_chambolle(imgDetails)
        
    def threshold(self, gray_image):
        """
        uses the otsu threshold method to generate a binary image

        Parameters:
        -----------
        gray_image: 2D array: gray scale image to be thresholded

        Return:
        --------
        2-D array of the binary image each pixel is either 1 or 0
        """
        thresholdValue = threshold_otsu(gray_image)
        return gray_image > thresholdValue
        
    def get_plate_like_objects(self):
        """
        uses principles of connected component analysis and labelling to map 
        out object regions.

        The plate dimensions were based on the following characteristics
        i.  They are rectangular in shape.
        ii. The width is more than the height
        iii. The ratio of the width to height is approximately 2:1
        iv. The proportion of the width of the license plate region to the 
        full image ranges between 15% to 40% depending on how the car image 
        was taken
        v.  The proportion of the height of the license plate region to the 
        full image is between 8% to 20%

        Return:
        --------
        3-D Array of license plate candidates region

        """
        self.label_image = measure.label(self.binary_image)
        self.plate_objects_cordinates = []
        threshold = self.binary_image
        plate_dimensions = (0.08*threshold.shape[0], 0.2*threshold.shape[0], 0.15*threshold.shape[1], 0.4*threshold.shape[1])
        minHeight, maxHeight, minWidth, maxWidth = plate_dimensions
        plate_like_objects = []
        for region in regionprops(self.label_image):
            if region.area < 10:
                continue
        
            minimumRow, minimumCol, maximumRow, maximumCol = region.bbox
            regionHeight = maximumRow - minimumRow
            regionWidth = maximumCol - minimumCol
            if regionHeight >= minHeight and regionHeight <= maxHeight and regionWidth >= minWidth and regionWidth <= maxWidth and regionWidth > regionHeight:
                plate_like_objects.append(self.full_car_image[minimumRow:maximumRow,
                    minimumCol:maximumCol])
                self.plate_objects_cordinates.append((minimumRow, minimumCol,
                    maximumRow, maximumCol))
                
        return plate_like_objects

    def validate_plate(self, candidates):
        """
        validates the candidate plate objects by using the idea
        of vertical projection to calculate the sum of pixels across
        each column and then find the average.

        This method still needs improvement

        Parameters:
        ------------
        candidate: 3D Array containing 2D arrays of objects that looks
        like license plate

        Returns:
        --------
        a 2D array of the likely license plate region

        """
        for each_candidate in candidates:
            height, width = each_candidate.shape
            each_candidate = self.inverted_threshold(each_candidate)
            license_plate = []
            highest_average = 0
            total_white_pixels = 0
            for column in range(width):
                total_white_pixels += sum(each_candidate[:, column])
            
            average = float(total_white_pixels) / width
            if average >= highest_average:
                license_plate = each_candidate

        return license_plate

    def inverted_threshold(self, grayscale_image):
        """
        used to invert the threshold of the candidate regions of the plate
        localization process. The inversion was neccessary
        because the license plate area is white dominated which means
        they have a greater gray scale value than the character region

        Parameters:
        -----------
        grayscale_image: 2D array of the gray scale image of the
        candidate region

        Returns:
        --------
        a 2D binary image
        """
        threshold_value = threshold_otsu(grayscale_image) - 0.05
        return grayscale_image < threshold_value

    def resize_if_necessary(self, image_to_resize):
        """
        function is used to resize the image before further
        processing if the image is too big. The resize is done
        in such a way that the aspect ratio is still maintained

        Parameters:
        ------------
        image_to_resize: 2D-Array of the image to be resized
        3D array image (RGB channel) can also be resized

        Return:
        --------
        resized image or the original image if resize is not
        neccessary
        """
        height, width = image_to_resize.shape
        ratio = float(width) / height
        # if the image is too big, resize
        if width > 600:
            width = 600
            height = round(width / ratio)
            return resize(image_to_resize, (height, width))

        return image_to_resize

class TemplateMatching:

    def template_match(self, predicted_label, image_data, training_dir):
        """
        applies the concept of template matching to determine the
        character among the similar ones that have the highest match and
        returns the label

        Parameters:
        ------------
        predicted_label: str; the character that was predicted by the machine
            learning model
        image_data: 2D numpy array image of the character that was predicted
        training_dir: the directory for the images that will be used in matching

        Returns:
        ---------
        The label with the highest match value
        """
        image_data = image_data.reshape(20, 20)
        prediction_fraction = self.fraction_match(predicted_label, training_dir,image_data)
        highest_fraction = prediction_fraction
        highest_fraction_label = predicted_label
        similar_labels_list = similar_characters[predicted_label]

        for each_similar_label in similar_labels_list:
            match_value = self.fraction_match(each_similar_label, training_dir,image_data)
            if match_value > highest_fraction:
                highest_fraction = match_value
                highest_fraction_label = each_similar_label

        return highest_fraction_label    

    def fraction_match(self, label, training_dir, image_data):
        fraction = 0
        extensions = {".jpg", ".png"}
        image_dir = os.path.join(training_dir, label)
        image_files = [f for f in os.listdir(image_dir) if (f.endswith(ext) for ext in extensions)] 
        for image_file in image_files:
            image_sample = imread(os.path.join(image_dir,image_file), as_gray=True)
            image_sample = image_sample < threshold_otsu(image_sample)
            match_fraction = match_template(image_data, image_sample)
            fraction += (match_fraction[0, 0] / len(image_files))
        return fraction


class TextClassification:
    
    def get_text(self, machine_learning_result):
        """
        combines the text classification from the machine machine learning 
        model

        Parameters:
        -----------
        machine_learning_result: 2D array of the machine learning
        model classification 

        Returns:
        --------
        string of the license plate but not in the right positioning
        """
        plate_string = ''
        for eachPredict in machine_learning_result:
            plate_string += eachPredict[0]
            
        return plate_string
    
    def text_reconstruction(self, plate_string, position_list):
        """
        returns the plate characters in the right order by using
        the starting columns of the character region

        Parameters:
        -----------
        plate_string: str; the license plate string in scatterred manner 
        position_list: 1D array of the starting columns of the character
        region

        Returns:
        --------
        String; the correctly ordered license plate text
        """
        posListCopy = position_list[:]
        position_list.sort()
        rightplate_string = ''
        for each in position_list:
            rightplate_string += plate_string[posListCopy.index(each)]
            
        return rightplate_string
