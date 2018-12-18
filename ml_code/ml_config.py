import os
import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.feature import match_template
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from skimage import restoration

# characters that should be clearly examined using template matching
confusing_chars = {'2', 'Z', 'B', '8', 'D', '0', '5', 'S', 'Q', 'R', '7'}
# a dictionary that keeps track of characters that are similar to the
# confusing characters
similar_characters = {
    '2':['Z'], 'Z':['2', '7'], '8':['B'], 'B':['8', 'R'], '5':['S'], 'S':['5'],
    '0':['D', 'Q'], 'D':['0', 'Q'], 'Q':['D', '0'], '7':['Z']
}

class MachineLearningConfig():
    def __init__(self):

        self.root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        training_20X20_dir = os.path.join(self.root_directory, 'media', 'training_data', 'train20X20')
        training_10X20_dir = os.path.join(self.root_directory, 'media', 'training_data', 'train10X20')
        self.training_data = [training_10X20_dir, training_20X20_dir]
        self.ascertain_characters = {'2', 'Z', 'B', '8', 'D', '0', '5', 'S', 'Q', 'R', '7'}

    def read_training_data(self, training_directory):
        """
        Reads each of the training data, thresholds it and appends it
        to a List that is converted to numpy array

        Parameters:
        -----------
        training_directory: str; of the training directory

        Returns:
        --------
        a tuple containing
        0: 2D numpy array of the training data with its features in 1D
        1: 1D numpy array of the labels (classifications)
        """

        image_data = []
        target_data = []
        extensions = {".jpg", ".png"}
        for each_letter in range(0,10) + [chr(x) for x in range(ord('A'), ord('Z')+1) if x != ord('I') | x != ord('O') ]:
            image_files = [f for f in os.listdir(os.path.join(training_directory, str(each_letter))) if (f.endswith(ext) for ext in extensions)] 
            for image_file in image_files:
                image_path = os.path.join(os.path.join(training_directory, str(each_letter)), image_file)
                image_details = imread(image_path, as_gray=True)
                image_details = restoration.denoise_tv_chambolle(image_details, weight=0.1)
                letter_details = image_details < threshold_otsu(image_details)
                binary_img = np.reshape(letter_details, -1)
                image_data.append(binary_img)
                target_data.append(str(each_letter))

        return (np.array(image_data), np.array(target_data))


    def save_model(self, model, foldername):
        """
        saves a model for later re-use without running the training 
        process all over again. Similar to how pickle works

        Parameters:
        -----------
        model: the machine learning model object
        foldername: str; of the folder to save the model
        """
        save_directory = os.path.join(self.root_directory, 'ml_models/'+foldername+'/')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        joblib.dump(model, os.path.join(save_directory, foldername+'.pkl'))

    def dimension_reduction(self, train_data, number_of_components):
        pca = PCA(number_of_components)
        return pca.fit_transform(train_data)

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
