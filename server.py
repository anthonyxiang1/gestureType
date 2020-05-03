'''
TO RUN:
$env:FLASK_APP = "server.py"
flask run

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json
import numpy as np
from scipy.interpolate import interp1d
import math
from operator import itemgetter 

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    #ediff1d gives difference between consecutive elements of the array
    #we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    #basically when words like mm or ii have no path / little path, use centroid
    if(distance[-1]==0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        #get the proportion of line segments
        distance = distance / distance[-1]
        #scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        #generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        #use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()
    
    return sample_points_X, sample_points_Y

def normalizeSamples(sample_X, sample_Y):
    L=1
    sample_Xnorm, sample_Ynorm = [], []

    width = max(sample_X) - min(sample_X)
    height = max(sample_Y) - min(sample_Y)

    if (width == 0 and height == 0):
        s = 0
    else:
        s = L / (max(width, height))
    
    sample_Xnorm = [s*x for x in sample_X]
    sample_Ynorm = [s*y for y in sample_Y]

    centerX = sum(sample_Xnorm)/len(sample_Xnorm)
    centerY = sum(sample_Ynorm)/len(sample_Ynorm)

    sample_Xnorm = [x - centerX for x in sample_Xnorm]
    sample_Ynorm = [y - centerY for y in sample_Ynorm]

    return sample_Xnorm, sample_Ynorm

# Pre-sample every template and get normalized templates
template_sample_points_X, template_sample_points_Y = [], []
template_sample_points_Xnorm, template_sample_points_Ynorm = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)
    Xnorm, Ynorm = normalizeSamples(template_sample_points_X[i], template_sample_points_Y[i])
    template_sample_points_Xnorm.append(Xnorm)
    template_sample_points_Ynorm.append(Ynorm)

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y, template_sample_points_Xnorm, template_sample_points_Ynorm):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10d000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).

    ADDED THESE
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Xnorm: 2D list, containing normalized X-axis values of every template (10d000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Ynorm: 2D list, containing normalized Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.

        I ADDED THESE 
        valid_template_sample_points_Xnorm: 2D list, the corresponding normalized X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Ynorm: 2D list, the corresponding normalized Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    gesture_Xnorm, gesture_Ynorm = normalizeSamples(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm = [], [], [], [], []
    # TODO: Set your own pruning threshold
    threshold = 0.15

    # TODO: Do pruning (12 points)
    for i in range(10000):
        if (abs(template_sample_points_Xnorm[i][0] - gesture_Xnorm[0]) < threshold and abs(template_sample_points_Ynorm[i][0] - gesture_Ynorm[0]) < threshold):   # compare first step
            if (abs(template_sample_points_Xnorm[i][-1] - gesture_Xnorm[-1]) < threshold and abs(template_sample_points_Ynorm[i][-1] - gesture_Ynorm[-1]) < threshold):   # compare last step
                valid_words.append(words[i])
                valid_template_sample_points_X.append(template_sample_points_X[i])  # these are not normalized
                valid_template_sample_points_Y.append(template_sample_points_Y[i])  # these are not normalized
                valid_template_sample_points_Xnorm.append(template_sample_points_Xnorm[i])  # these are normalized
                valid_template_sample_points_Ynorm.append(template_sample_points_Ynorm[i])  # these are normalized

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.

    I CHANGED THESE PARAMETERS - SINCE WE ARE USING NORMALIZED VALUES
    :param valid_template_sample_points_Xnorm: 2D list, containing normalized X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Ynorm: 2D list, containing normalized Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''

    gesture_Xnorm, gesture_Ynorm = normalizeSamples(gesture_sample_points_X, gesture_sample_points_Y)

    shape_scores = [0]*len(valid_template_sample_points_Xnorm)

    # TODO: Calculate shape scores (12 points)
    for i in range(len(valid_template_sample_points_Xnorm)):
        for j in range(100):
            shape_scores[i] += math.sqrt((gesture_Xnorm[j] - valid_template_sample_points_Xnorm[i][j])**2 + (gesture_Ynorm[j] - valid_template_sample_points_Ynorm[i][j])**2)/100

    return shape_scores

def isNotZeroD(gestureX, gestureY, templateSampleX, templateSampleY):
    radius = 30

    for i in range(100):
        d = math.sqrt((min(gestureX, key=lambda x:abs(x-templateSampleX[i])))**2 + (min(gestureY, key=lambda y:abs(y-templateSampleY[i])))**2)
        if ( d - radius > 0):
            return True

    return False

def getBeta(gestureX, gestureY, templateSampleX, templateSampleY, sampleNum):
    if (not isNotZeroD(gestureX, gestureY, templateSampleX, templateSampleY) and not isNotZeroD(templateSampleX, templateSampleY, gestureX, gestureY)):
        return 0
    
    return math.sqrt((gestureX[sampleNum] - templateSampleX[sampleNum])**2 + (gestureY[sampleNum] - templateSampleY[sampleNum])**2)

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    
    alpha = [0] * 100
    for i in range(100):
        alpha[i] = (math.ceil(abs(50.5 - i-1)))/2550

    location_scores = [0] * len(valid_template_sample_points_X)
    
    # TODO: Calculate location scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        temp = 0
        templateX = valid_template_sample_points_X[i]
        templateY = valid_template_sample_points_Y[i]
        for j in range(100):
            temp += alpha[j]*getBeta(gesture_sample_points_X, gesture_sample_points_Y, templateX, templateY, j)/100
        location_scores[i] = temp

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.5
    # TODO: Set your own location weight
    location_coef = 1 - shape_coef
    for i in range(len(location_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores

def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = ''
    final_score = float('inf')
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)

    dictionary = dict(zip(valid_words, integration_scores))
    filtered = dict(sorted(dictionary.items(), key = itemgetter(1))[:n])
    #print(filtered)

    for word, int_score in filtered.items():
        if (int_score*(1-probabilities[word]) < final_score):
            final_score = int_score*(1-probabilities[word])
            best_word = word

    if (best_word == ''):
        return "No best word found"

    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y, template_sample_points_Xnorm, template_sample_points_Ynorm)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_Xnorm, valid_template_sample_points_Ynorm)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"'+ best_word  + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'

if __name__ == "__main__":
    app.run()
