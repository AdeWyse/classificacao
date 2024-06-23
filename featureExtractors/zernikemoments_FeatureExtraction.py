import os
import cv2
import numpy as np
from sklearn import preprocessing
from progress.bar import Bar
import time
import mahotas as mh

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/zernikemoments/train/'
    testFeaturePath = './features_labels/zernikemoments/test/'
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractZernikeMomentsFeatures(trainImages)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractZernikeMomentsFeatures(testImages)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 0:  # it's inside a folder with files
                folder_name = os.path.basename(dirpath)
                bar = Bar(f'[INFO] Getting images and labels from {folder_name}', max=len(filenames), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels, dtype=object)
    
def extractZernikeMomentsFeatures(images):
    bar = Bar('[INFO] Extracting Zernike moments features...', max=len(images), suffix='%(index)d/%(max)d Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if len(image.shape) > 2: # checa numero de canais, se maior que dois precisa converter
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converte para grascale
        blur_image = cv2.medianBlur(image, 3) # Redução de ruido
        _, binary_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #separa formas usando otsu
        radius = min(binary_image.shape) // 2 - 1 # Calcula a região onde vai ser aplicado o zernike moments. radius maiores para grandes e radius menores para areas pequenas com mais detalhe
        features = mh.features.zernike_moments(binary_image, radius) #encontra caracteristica utilizando zernike moments
        featuresList.append(features)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = f'{labels=}'.split('=')[0] + '.csv'
    feature_filename = f'{features=}'.split('=')[0] + '.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0] + '.csv'
    os.makedirs(path, exist_ok=True)
    np.savetxt(path + label_filename, labels, delimiter=',', fmt='%i')
    np.savetxt(path + feature_filename, features, delimiter=',')
    np.savetxt(path + encoder_filename, encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
