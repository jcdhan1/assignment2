"""Jack Cheng Ding Han
  150159519
  ACA15JCH"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import types
import operator
from scipy import linalg

GRID_LENGTH=15
BLOCK_LENGTH=30

data = pickle.load(open("assignment2.pkl", "rb"))
train_data = data['train_data']
train_labels = data['train_labels']
test1 = data['test1']
test2 = data['test2']
words  = data['words']

#This is used to measure how much the classify function gets correct on test1 or test2.
TEST_LABELS = np.array([14,18,25, 5,12, 7,14, 1,12,14,12, 5,16, 2,18,
                        21,14,15,14, 6, 8, 2,18,15,16, 5,18,18,13,16,
                         3,15,14, 2,20, 8, 7, 9,18,23,25, 2,14, 1, 6,
                         1,20,23, 1, 9, 5, 5,23,15,18, 3, 1, 9,12,22,
                        14,16,15, 6, 4,14, 5,15,23, 5,13,14,15, 1,10,
                         5, 5,18,11, 9,18,19, 1,20, 5,5 ,23,14, 5,16,
                        12,18, 2,14, 5,19, 8,15, 7,23, 5, 2,12, 5, 1,
                        12,15, 4, 5,18,19, 8, 4,14,18,18,12,16,14,24,
                        25,20, 9,12, 4,18, 9,25, 4,21, 9, 8,25,15,20,
                        11, 5,18,18, 5,18,16, 5, 7, 3,24,15,18,20,15,
                         5,16, 1,14, 2, 9,23, 8,15, 6,16, 1,18,19,14,
                        10, 5, 5,14, 9,15, 6, 5, 8,14,18,18, 1,14,26,
                         2, 9, 4,14,14,15, 4,19, 5,11,15, 5, 2, 5,10,
                        14, 2,18,20,18,19,12,14, 5, 1, 8, 1, 7, 8,14,
                         1,12, 5,21,20,14, 5,11,23,14, 9,12, 5,19,12])

#The starts and ends of each line drawn by wordsearch in each will be compared to this.
CORRECT_LINES = [(192,132),(180, 68),(154, 42),( 92, 32),( 30, 75),( 55, 51),(186,184),
                 ( 63,111),( 44,156),(131,191),(165, 90),( 74,172),(217,214),(  8,  2),
                 (219,107),( 29, 85),( 89,164),(151,106),( 91, 16),(  1,113),( 22, 26),
                 (223,103),(59, 157),( 39, 34)]

def get_name(testN):
    """Name of test data.
    testN - test1 or test2

    returns: name of the test data as a string.
    """
    return "test1" if np.ndarray.tolist(test1)==np.ndarray.tolist(testN) else "test2"

def get_characters(ndimarray):
    """This allows the extraction of individual character images.

    ndimarray - the matrix of pixels for every character

    returns: For m n by n matrices, this returns a m by o matrix where o=n*n.
    """
    letters = []
    cols = (ndimarray.shape[1])
    rows = (ndimarray.shape[0])
    col_step = cols//GRID_LENGTH
    row_step = rows//GRID_LENGTH
    for current_row_start in range(0, rows, row_step):
        for slice_start in range(0,cols,col_step):
            letter = []
            for i in range(current_row_start,current_row_start+row_step):
                row_of_single_letter = ndimarray[i][slice_start:slice_start+col_step]
                letter+=[row_of_single_letter]
            letters+=[letter]
    return np.array([np.reshape(np.transpose(feature_vector),
                    (900,), order='C') for feature_vector in np.array(letters)])

#Classification
def classify(train, train_labels, test):
    """Nearest neighbour classification

    train - matrix of training data (one sample per row)
    train_labels - corresponding training data labels
    test - matrix of samples to classify

    returns: labels - vector of test data labels
    """
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test*test, axis=1))
    modtrain = np.sqrt(np.sum(train*train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest = np.argmax(dist, axis=1)
    labels = train_labels[nearest]
    return labels

def correctly_processed(classified,actual,percentage):
    """Compares the results of the classifier to the actual answer
    classified - the data that's already been classified
    actual - the actual answer
    percentage - if this is True, a percentage is returned.
    
    returns: Number or percentage correct.
    """
    correct=0
    for i in range(0,classified.shape[0]):
        if (classified[i]==actual[i]):
            correct+=1
    return int(correct/(0.01*classified.shape[0] if percentage else 1))

def search(list_to_search,word,direction='A'):
    """This returns a list of tuples. In each tuple, the first value is the index of
    where the word could begin, the second is the index of where it could end, and the
    third is the number of characters that match the word to search with. To work fast
    fast when there are exact matches found, they are immediately returned.

    ist_to_search - list of tuples. In each tuple, the first value is the original index
                    of the label, the second is the label found using classify.
    word - word to search with
    direction - direction to search in. It can be 'F' for forward, 'B' for back or 'A'
                for all.

    returns: (0,0,0) with the exact match if such was found, list_to_search with more
             tuples if inexact matches were found and list_to_search is no new matches
             were found.
    """
    if (len(word)>len(list_to_search)):
        return []
    unzipped=list(zip(*list_to_search))
    orig_indices=list(unzipped[0])
    if (direction=='F'): #Search forwards
        arr_to_search=np.array(unzipped[1])
        pos=0
        forward=[]
        while len(word)<=len(arr_to_search[pos:]):
            num_correct = sum(word==(arr_to_search[pos:pos+len(word)]))
            #If an exact match is found, return it immediately with (0,0,0).
            if num_correct==len(word):
                return [(0,0,0),(orig_indices[pos],
                        orig_indices[pos+len(word)-1],num_correct)]
            #Ignore totally incorrect tuples to increase sorting speed later on.
            if num_correct>0:
                forward+=[(orig_indices[pos],orig_indices[pos+len(word)-1],num_correct)]
            pos+=1
        return forward
    elif (direction=='B'): #Search backwards
        reversed_list=list(list_to_search)
        reversed_list.reverse()
        return search(reversed_list,word,'F')
    else: #Search in all directions and combine
        return search(list_to_search,word,'F') + search(list_to_search,word,'B')

def i_label_list(grid_to_search,row_col_diag):
    """Combines rows, columns or diagonals of the indices grid with the respective labels
    in the classified matrix.

    grid_to_search - a matrix already been classified
    row_col_diag - a row, column or diagonal of a matrix of indices.

    returns: A list of tuples.
    """
    return list(zip(row_col_diag,[grid_to_search[indx] for indx in row_col_diag]))

def trav_diag(grid_to_search,i_grid, l_searched, w_to_search, per=False):
    """This function is for searching all diagonals.

    i_grid - The grid of original indices.
    l_searched - The current lists searched.
    w_to_search - word with characters converted to label numbers
    per - Search through perpendicular diagonals if this is True.

    returns: A list of tuples.
    """
    indices_g=np.fliplr(i_grid) if per else i_grid
    journeys=0
    diag_n=0
    while journeys<2:
        par_diag=np.diagonal(indices_g,diag_n)
        new_tuples=search(i_label_list(grid_to_search,par_diag),w_to_search)
        #Return immediately if exact match found.
        if len(new_tuples)>0 and new_tuples[0]==(0,0,0):
            return new_tuples
        l_searched+=new_tuples
        if len(w_to_search)<len(par_diag):
            diag_n+=1-2*journeys
        else:
            if len(w_to_search)==len(par_diag):
                diag_n=-1
            journeys+=1
    if not per: #After parallel diagonals, search through perpendicular diagonals.
        per_list=trav_diag(grid_to_search,i_grid,l_searched,w_to_search,"per")
        #if per_list has an exact match, don't combine with l_searched
        if len(per_list)>0 and per_list[0]==(0,0,0):
            return per_list
        l_searched+=per_list
    return l_searched

def draw_line(graph, start, end, side_length):
    """This draws a line on a pyplot between two letters.

    graph - the pyplot to draw on
    start_letter - where the line begins, can be a coordinate or index.
    end_letter - where the line ends, can be a coordinate or index.
    """
    if (isinstance(start, int)):
        draw_line(graph, [start % GRID_LENGTH,start//GRID_LENGTH],
                  [end % GRID_LENGTH,end//GRID_LENGTH], side_length)
    else:
        xSxE=[start[0]*side_length+GRID_LENGTH,end[0]*side_length+GRID_LENGTH]
        ySyE=[start[1]*side_length+GRID_LENGTH,end[1]*side_length+GRID_LENGTH]
        plt.xlim(0, GRID_LENGTH*side_length)
        plt.ylim(GRID_LENGTH*side_length, 0)
        graph.plot(xSxE, ySyE, 'y-',linewidth=2) #Plot a yellow line on the graph

def reduce(test_dat,train_dat,n,pca_indices=range):
    """Using principle component analysis to reduce 900 features down to 10 features.

    test_data - Testing data to reduce
    train_data - Training data to use
    n - The number of features to be returned.
    pca_indices - The indices of the features. range(n) if unspecified.

    returns: For a p by q matrix, a p by n matrix is returned.
    """
    pca_i=pca_indices(n) if type(pca_indices) is type else pca_indices
    covx = np.cov(train_data, rowvar=0)
    N_Orig = covx.shape[0]
    w, v = linalg.eigh(covx, eigvals=(N_Orig-(pca_i[len(pca_i)-1]+1), N_Orig-1))
    v = np.fliplr(v)
    features=np.dot((test_dat - np.mean(train_dat)), v) 
    return np.array([[vec[k] for k in pca_i]for vec in features])

def wordsearch(testN, words_to_find, train_dat, train_lbl,reduced=False,pca_i=range(10)):
    """This is the main wordsearch function.

    testN - The test data e.g. test1 or test2
    words_to_find - words
    train_dat - The training data
    train_lbl - The labels of the training data
    reduced - If this is true, the testing data is reduced down to 10 features.
    pca_i = The indices of the features to use, if empty, assume 0 up to and including 9.
    """
    print("Solving " + get_name(testN) + (" With Reduction" if reduced else ""))
    plt.matshow(testN, cmap=cm.gray)
    #Classification, Pre-processing, Dimensionality Reduction
    training_dat = reduce(train_dat,train_dat,10,pca_i) if reduced else train_dat
    preprocessed = get_characters(testN)
    if reduced:
        classified_mat = classify(training_dat, train_lbl,
                                  reduce(preprocessed,train_dat,10,pca_i))
    else:
        classified_mat = classify(training_dat, train_lbl, preprocessed)
    print(str(correctly_processed(classified_mat,TEST_LABELS,False))
      + " letters were correctly labelled which is about "
      + str(correctly_processed(classified_mat,TEST_LABELS,True)) + "% of the letters.")
    """Give every element a unique number because the functions to be used such as
       diagonal do not preserve indices."""
    indices_grid=np.reshape(range(classified_mat.shape[0]), (GRID_LENGTH,GRID_LENGTH))
    #Select a word to find
    lines=[]
    for w_string in words_to_find:
        w=[ord(ch)-96 for ch in list(w_string)]
        #Perform the search
        print("Searching for " + w_string)
        lists_searched = []
        for r in range(0,GRID_LENGTH): #Search row by row
            lists_searched+=search(i_label_list(classified_mat,indices_grid[r]),w)
        if lists_searched[0]!=(0,0,0): #Search column by column if no exact match yet.
            for c in range(0,GRID_LENGTH):
                lists_searched+=search(i_label_list(classified_mat,indices_grid[:,c]),w)
        if lists_searched[0]!=(0,0,0): #Search all diagonals if no exact match yet
            lists_searched=trav_diag(classified_mat,indices_grid,lists_searched,w)
        if lists_searched[0]==(0,0,0): #Exact matches at index [1].
            likeliest=lists_searched[1]
        else: #Otherwise, the likeliest is the tuple that has the biggest third element.
            likeliest=sorted(lists_searched,key=lambda t: t[2])[len(lists_searched)-1]
        likeliest_tuple=likeliest[:2]
        lines+=[likeliest_tuple]
        #Display the result
        print("Line from index "
              + str(likeliest_tuple[0]) + " to index " + str(likeliest_tuple[1]))
        draw_line(plt,int(likeliest_tuple[0]), int(likeliest_tuple[1]), BLOCK_LENGTH)
    print("Solved")
    print(str(sum([t[0] and t[1] for t in (np.array(CORRECT_LINES)==(np.array(lines)))]))
          + " out of " + str(len(words)) + " found correctly.")
    plt.gcf().savefig(get_name(testN) + ("_reduced" if reduced else ""), dpi=100)
PCA_I=[1, 2, 3, 5, 6, 7, 8, 9, 10, 11] #Indices of features selected through PCA.
wordsearch(test1, words, train_data, train_labels) #Trial1
wordsearch(test1, words, train_data, train_labels,True,PCA_I) #Trial2
wordsearch(test2, words, train_data, train_labels,True,PCA_I) #Trial3
