import pandas as pd
import matplotlib.pyplot as plt

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