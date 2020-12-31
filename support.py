import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

def Plot_digit(row):
    first_digit = row.reshape(28, 28)
    plt.imshow(first_digit, cmap='binary')
    plt.axis("off")
    plt.show()

def Test(y_pred, y_answers):
    correct = sum(y_pred == y_answers)
    return correct/len(y_pred)

def Read_in_data(path, test=False):
    data_as_csv = pd.read_csv(path)
    if test:
        return data_as_csv.iloc[:].to_numpy(), data_as_csv.iloc[0:, 0].to_numpy()
    return data_as_csv.iloc[1:, 1:].to_numpy(), data_as_csv.iloc[1:, 0].to_numpy()

def predictions_to_submission(name, predictions):
    eval_results_file = open(name + '.txt', "w")
    eval_results_file.writelines('ImageId,Label\n')

    for i in range(len(predictions)):
        eval_results_file.writelines(str(i+1) + ',' + str(predictions[i]) + '\n')
    eval_results_file.close()

def skelefy(X):
    toRet = []
    for x in X:
        ret, img = cv2.threshold(x.reshape(28, 28, 1).astype(np.uint8), 127, 255, 0)
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        toRet.append(skel)
        # toRet.append(skel.flatten())
    return toRet

def join_count(X):
    toRet = []
    for x in X:
        im_or = x.copy()
        x.astype(np.int32)
        # create kernel
        kernel = np.ones((7, 7))
        kernel[2:5, 2:5] = 0
        print(kernel)
        # apply kernel
        res = cv2.filter2D(x, 3, kernel)
        # filter results
        loc = np.where(res > 2800)
        print(len(loc[0]))
        # draw circles on found locations
        for j in range(len(loc[0])):
            cv2.circle(im_or, (loc[1][j], loc[0][j]), 10, (127), 5)
        # display result
        cv2.imshow('Result', im_or)
        cv2.waitKey(0)
        cv2.destroyAllWindows()