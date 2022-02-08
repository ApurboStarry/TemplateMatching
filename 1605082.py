import cv2
import numpy as np


class TemplateMatching:
    def __init__(self, videoFileName, imageFileName):
        self.videoFileName = videoFileName
        self.imageFileName = imageFileName

        self.videocapture, self.originalFrames, self.grayscaleFrames = self.getVideoFrames()
        self.grayscaleFrameHeight, self.grayscaleFrameWidth = self.grayscaleFrames[0].shape

        self.referenceImage = self.getReferenceImage()
        self.referenceImageHeight, self.referenceImageWidth = self.referenceImage.shape

    def exhaustiveSearchUtil(self, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        minimum_value = np.inf
        new_best_location = -1, -1

        search_counter = 0

        for i in range(previous_best_height - p, previous_best_height + p + 1):
            for j in range(previous_best_width - p, previous_best_width + p + 1):
                isValidLocation = i >= 0 and i + self.referenceImageHeight < grayscale_frame_matrix.shape[0] and j >= 0 and j + self.referenceImageWidth < grayscale_frame_matrix.shape[1]
                if not isValidLocation:
                    continue
                search_counter += 1
                block_matrix = grayscale_frame_matrix[i: i +
                                                      self.referenceImageHeight, j: j + self.referenceImageWidth]
                value = np.sum((self.referenceImage / 255.0 - block_matrix / 255.0) ** 2)
                if value < minimum_value:
                    minimum_value = value
                    new_best_location = i, j

        return new_best_location, search_counter

    def exhaustiveSearch(self, p):
        frameMatrices = self.originalFrames.copy()

        # exhaustive search on entire matrix
        best_location = -1, -1
        minimum_value = np.inf

        firstGrayScaleFrame = self.grayscaleFrames[0]

        search_counter = 0
        for i in range(self.grayscaleFrameHeight - self.referenceImageHeight + 1):
            for j in range(self.grayscaleFrameWidth - self.referenceImageWidth + 1):
                search_counter += 1
                block_matrix = firstGrayScaleFrame[i: i +
                                                   self.referenceImageHeight, j: j + self.referenceImageWidth]
                value = np.sum((self.referenceImage / 255.0 - block_matrix / 255.0) ** 2)
                if value < minimum_value:
                    minimum_value = value
                    best_location = i, j

        total_search_counter = search_counter
        for i in range(1, len(frameMatrices)):
            best_location_top_left, search_counter = self.exhaustiveSearchUtil(
                self.grayscaleFrames[i], best_location, p)

            best_location_top_left = best_location_top_left[::-1]
            
            best_location_bottom_right = best_location_top_left[0] + \
            self.referenceImageWidth, best_location_top_left[1] + \
            self.referenceImageHeight
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frameMatrices[i], best_location_top_left,
                        best_location_bottom_right, color, thickness)
            
            best_location = best_location_top_left[::-1]
            total_search_counter += search_counter

        # generating video
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        height, width, layers = frameMatrices[0].shape
        outputV = cv2.VideoWriter(
            'exhaustive - output.mov', fourcc, fps, (width, height))
        for frame in frameMatrices:
            outputV.write(frame)
        outputV.release()

        return total_search_counter

    def logarithmicSearchUtil(self, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        minimum_value = np.inf
        new_best_location = -1, -1

        k = int(np.ceil(np.log2(p)))
        d = 2 ** (k - 1)
        p //= 2

        search_counter = 0

        while d > 1:
            for i in range(previous_best_height - d, previous_best_height + d + 1, d):
                for j in range(previous_best_width - d, previous_best_width + d + 1, d):
                    isValidLocation = i >= 0 and i + self.referenceImageHeight < grayscale_frame_matrix.shape[0] and j >= 0 and j + self.referenceImageWidth < grayscale_frame_matrix.shape[1]
                    if not isValidLocation:
                        continue
                    search_counter += 1
                    block_matrix = grayscale_frame_matrix[i: i +
                                                          self.referenceImageHeight, j: j + self.referenceImageWidth]
                    value = np.sum((self.referenceImage / 255.0 - block_matrix / 255.0) ** 2)
                    if value < minimum_value:
                        minimum_value = value
                        new_best_location = i, j
            k = int(np.ceil(np.log2(p)))
            d = 2 ** (k - 1)
            p //= 2

        return new_best_location, search_counter

    def logarithmicSearch(self, p):
        frameMatrices = self.originalFrames.copy()

        # exhaustive search on entire matrix
        best_location = -1, -1
        minimum_value = np.inf

        firstGrayScaleFrame = self.grayscaleFrames[0]

        search_counter = 0
        for i in range(self.grayscaleFrameHeight - self.referenceImageHeight + 1):
            for j in range(self.grayscaleFrameWidth - self.referenceImageWidth + 1):
                search_counter += 1
                block_matrix = firstGrayScaleFrame[i: i +
                                                   self.referenceImageHeight, j: j + self.referenceImageWidth]
                value = np.sum((self.referenceImage / 255.0 - block_matrix / 255.0) ** 2)
                if value < minimum_value:
                    minimum_value = value
                    best_location = i, j

        total_search_counter = search_counter
        for i in range(1, len(frameMatrices)):
            best_location_top_left, search_counter = self.logarithmicSearchUtil(
                self.grayscaleFrames[i], best_location, p)

            best_location_top_left = best_location_top_left[::-1]
            
            best_location_bottom_right = best_location_top_left[0] + \
                self.referenceImageWidth, best_location_top_left[1] + \
                self.referenceImageHeight
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frameMatrices[i], best_location_top_left,
                          best_location_bottom_right, color, thickness)
            
            best_location = best_location_top_left[::-1]
            total_search_counter += search_counter

        # generating video
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        height, width, layers = frameMatrices[0].shape
        outputV = cv2.VideoWriter(
            'logarithmic - output.mov', fourcc, fps, (width, height))
        for frame in frameMatrices:
            outputV.write(frame)
        outputV.release()

        return total_search_counter

    def submatrixSearch(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        reference_image_matrix_height, reference_image_matrix_width = reference_image_matrix.shape
        minimum_value = np.inf
        new_best_location = -1, -1

        search_counter = 0

        for i in range(previous_best_height - p, previous_best_height + p + 1):
            for j in range(previous_best_width - p, previous_best_width + p + 1):
                reference_image_matrix_height, reference_image_matrix_width = reference_image_matrix.shape
                frame_matrix_height, frame_matrix_width = grayscale_frame_matrix.shape
                if not (i >= 0 and i + reference_image_matrix_height < frame_matrix_height and j >= 0 and j + reference_image_matrix_width < frame_matrix_width):
                    continue
                search_counter += 1
                block_matrix = grayscale_frame_matrix[i: i +
                                                      reference_image_matrix_height, j: j + reference_image_matrix_width]
                value = np.sum((reference_image_matrix / 255.0 - block_matrix / 255.0) ** 2)
                if value < minimum_value:
                    minimum_value = value
                    new_best_location = i, j

        return new_best_location, search_counter

    def hierarchicalSearchUtil(self, grayscale_frame_matrix, previous_best_location, p):
        level_wise_reference_image_matrices = [self.referenceImage]
        level_wise_reference_image_matrices.append(
            cv2.pyrDown(level_wise_reference_image_matrices[0]))
        level_wise_reference_image_matrices.append(
            cv2.pyrDown(level_wise_reference_image_matrices[1]))

        level_wise_grayscale_frame_matrices = [grayscale_frame_matrix]
        level_wise_grayscale_frame_matrices.append(
            cv2.pyrDown(level_wise_grayscale_frame_matrices[0]))
        level_wise_grayscale_frame_matrices.append(
            cv2.pyrDown(level_wise_grayscale_frame_matrices[1]))

        x, y = previous_best_location
        (x1, y1), counter1 = self.submatrixSearch(
            level_wise_reference_image_matrices[2], level_wise_grayscale_frame_matrices[2], (x // 4, y // 4), p // 4)
        (x2, y2), counter2 = self.submatrixSearch(
            level_wise_reference_image_matrices[1], level_wise_grayscale_frame_matrices[1], (2 * x1, 2 * y1), p // 2)
        best_location, counter3 = self.submatrixSearch(
            level_wise_reference_image_matrices[0], level_wise_grayscale_frame_matrices[0], (2 * x2, 2 * y2), p)

        search_counter = counter1 + counter2 + counter3

        return best_location, search_counter

    def hierarchicalSearch(self, p):
        frameMatrices = self.originalFrames.copy()

        # exhaustive search on entire matrix
        best_location = -1, -1
        minimum_value = np.inf

        firstGrayScaleFrame = self.grayscaleFrames[0]

        search_counter = 0
        for i in range(self.grayscaleFrameHeight - self.referenceImageHeight + 1):
            for j in range(self.grayscaleFrameWidth - self.referenceImageWidth + 1):
                search_counter += 1
                block_matrix = firstGrayScaleFrame[i: i +
                                                   self.referenceImageHeight, j: j + self.referenceImageWidth]
                value = np.sum((self.referenceImage / 255.0 - block_matrix / 255.0) ** 2)
                if value < minimum_value:
                    minimum_value = value
                    best_location = i, j

        total_search_counter = search_counter
        for i in range(1, len(frameMatrices)):
            best_location_top_left, search_counter = self.hierarchicalSearchUtil(
                self.grayscaleFrames[i], best_location, p)

            best_location_top_left = best_location_top_left[::-1]
            
            best_location_bottom_right = best_location_top_left[0] + \
                self.referenceImageWidth, best_location_top_left[1] + \
                self.referenceImageHeight
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frameMatrices[i], best_location_top_left,
                          best_location_bottom_right, color, thickness)
            
            best_location = best_location_top_left[::-1]
            total_search_counter += search_counter

        # genearating video
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        height, width, layers = frameMatrices[0].shape
        outputV = cv2.VideoWriter(
            'hierarchical - output.mov', fourcc, fps, (width, height))
        for frame in frameMatrices:
            outputV.write(frame)
        outputV.release()

        print(total_search_counter)
        return total_search_counter
      
      
    def prepareReport(self, exhaustiveArray, logarithmicArray, heirarchicalArray):
      file1 = open("1605082 Report.txt", "w")
      file1.write("p - exhaustive - logarithmic - hierarchical")
      file1.write("\n")
      
      for i in range(len(exhaustiveArray)):
        file1.write(str(exhaustiveArray[i][0]) + " - " + str(exhaustiveArray[i][1]) + " - " + str(logarithmicArray[i][1]) + " - " + str(heirarchicalArray[i][1]))
        file1.write("\n")
        # print(exhaustiveArray[i][0], exhaustiveArray[i][1], logarithmicArray[i][1], heirarchicalArray[i][1])
      
      file1.close()

    def runOffline(self):
        exhaustive_list = []        
        for p in range(5, 11):
          counter = self.exhaustiveSearch(p)
          exhaustive_list.append((p, counter / len(self.originalFrames)))
        exhaustive_array = np.asarray(exhaustive_list)
        
        logarithmic_list = []
        for p in range(5, 11):
          counter = self.logarithmicSearch(p)
          logarithmic_list.append((p, counter / len(self.originalFrames)))
        logarithmicArray = np.asarray(logarithmic_list)
        
        hierarchical_list = []
        for p in range(5, 11):
          counter = self.hierarchicalSearch(p)
          hierarchical_list.append((p, counter / len(self.originalFrames)))
        hierarchicalArray = np.asarray(hierarchical_list)
        
        self.prepareReport(exhaustive_array, logarithmicArray, hierarchicalArray)

    def getReferenceImage(self):
        return cv2.cvtColor(cv2.imread(self.imageFileName), cv2.COLOR_BGR2GRAY)

    def getVideoFrames(self):
        original_frame_matrices = []
        grayscale_frame_matrices = []
        vidcap = cv2.VideoCapture(self.videoFileName)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            original_frame_matrices.append(image)
            grayscale_frame_matrices.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        return vidcap, np.asarray(original_frame_matrices), np.asarray(grayscale_frame_matrices)


tm = TemplateMatching("input.mov", "reference.jpg")
tm.runOffline()
