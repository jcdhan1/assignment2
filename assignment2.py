import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import types

#Pre-processing
GRID_LENGTH=15
BLOCK_LENGTH=30

data = pickle.load(open("assignment2.pkl", "rb"))
train_data = data['train_data']
train_labels = data['train_labels']
test1 = data['test1']
test2 = data['test2']
words  = data['words']
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

def get_name(testN):
	return "test1" if np.ndarray.tolist(test1)==np.ndarray.tolist(testN) else "test2"

def get_characters(ndimarray, as_vectors):
	"""This allows the extraction of individual character images as 30 by 30 matrices
	   ndimarray: the matrix of pixels for every character
	   as_vectors: if this is true, a 225 by 900-element matrix is returned."""
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
	if as_vectors:
		return np.array([np.reshape(np.transpose(feature_vector), (900,), order='C') for feature_vector in np.array(letters)])
	else:
		return np.array(letters)

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
	correct=0
	for i in range(0,classified.shape[0]):
		if (classified[i]==actual[i]):
			correct+=1
	return int(correct/(0.01*classified.shape[0] if percentage else 1))

#Perform the search
def search(list_to_search,word,direction='A'):
	"""This returns a list of tuples. In each tuple, the first value is the index of where the word could begin, the second is the index of
	   where it could end, and the third is the number of characters that match the word to search with.
	   list_to_search: list of tuples. In each tuple, the first value is the original index of the label, the second is the label found
					   using classify.
	   word: word to search with
	   direction: direction to search in. It can be 'F' for forward, 'B' for back or 'A' for all."""
	if (len(word)>len(list_to_search)):
		return []
	unzipped=list(zip(*list_to_search))
	orig_indices=list(unzipped[0])
	
	if (direction=='F'):
		#Search forwards
		arr_to_search=np.array(unzipped[1])
		pos=0
		forward=[]
		while len(word)<=len(arr_to_search[pos:]):
			num_correct = sum(word==(arr_to_search[pos:pos+len(word)]))
			#Increase sorting speed later on, ignore totally incorrect results, only include totally correct results if any.
			if num_correct>0:
				forward+=[(orig_indices[pos],orig_indices[pos+len(word)-1],num_correct)]
			elif num_correct==len(word):
				forward==[(orig_indices[pos],orig_indices[pos+len(word)-1],num_correct)]
			pos+=1
		return forward
	elif (direction=='B'):
		#Search backwards
		reversed_list=list(list_to_search)
		reversed_list.reverse()
		return search(reversed_list,word,'F')
	else:
		#Search in all directions and combine
		return search(list_to_search,word,'F') + search(list_to_search,word,'B')

def index_label_list(grid_to_search,row_col_diag):
	return list(zip(row_col_diag,[grid_to_search[indx] for indx in row_col_diag]))

def trav_diag(grid_to_search,i_grid, l_searched, w_to_search, orientation="par"):
	"""	 i_grid: The grid of original indices.
		l_searched: The current lists searched.
	   w_to_search: word with characters converted to label numbers
	   orientation: "par" for parallel to leading diagonal, "per" for perpendicular to leading diagonal."""
	
	"""Diagonals perpendicular to the leading diagonal of i_grid are equivalent to diagonals parallel to the leading diagonal
	   of the reflection of i_grid."""
	indices_g=np.fliplr(i_grid) if orientation=="per" else i_grid
	journeys=0
	diag_n=0
	while journeys<2:
		par_diag=np.diagonal(indices_g,diag_n)
		l_searched+=search(index_label_list(grid_to_search,par_diag),w_to_search)
		if len(w_to_search)<len(par_diag):
			diag_n+=1-2*journeys
		else:
			if len(w_to_search)==len(par_diag):
				diag_n=-1
			journeys+=1
	# Once parallel diagonals have been searched, search through perpendicular diagonals and add to the list.
	if orientation=="par":
		l_searched+=trav_diag(grid_to_search,i_grid,l_searched,w_to_search,"per")
	return l_searched

#Display the result
def draw_line(graph, start, end, side_length):
	"""This draws a line on a pyplot between two letters.
	   graph: the pyplot to draw on
	   start_letter: where the line begins, can be a coordinate or index.
	   end_letter: where the line ends, can be a coordinate or index."""
	
	if (isinstance(start, int)):
		draw_line(graph, [start % GRID_LENGTH,start//GRID_LENGTH], [end % GRID_LENGTH,end//GRID_LENGTH], side_length)
	else:
		# get starting x-coordinate, get ending x-coordinate.
		xSxE=[start[0]*side_length+GRID_LENGTH,end[0]*side_length+GRID_LENGTH]
		# get starting y-coordinate, get ending y-coordinate.
		ySyE=[start[1]*side_length+GRID_LENGTH,end[1]*side_length+GRID_LENGTH]
		# Plot a yellow line on the graph
		plt.xlim(0, GRID_LENGTH*side_length)
		plt.ylim(GRID_LENGTH*side_length, 0)
		graph.plot(xSxE, ySyE, 'y-')

def wordsearch(testN, words_to_find, train_dat, train_lbl):
	print("Solving " + get_name(testN))
	plt.matshow(testN, cmap=cm.gray)
	#Classification, Pre-processing
	classified_mat = classify(train_dat, train_lbl, get_characters(testN, True))
	print(str(correctly_processed(classified_mat,TEST_LABELS,False))
	  + " letters were correctly labeled which is about "
	  + str(correctly_processed(classified_mat,TEST_LABELS,True)) + "% of the letters.")
	#Give every element a unique number because the functions to be used such as diagonal do not preserve indices
	indices_grid=np.reshape(range(classified_mat.shape[0]), (GRID_LENGTH,GRID_LENGTH))
	#Select a word to find
	for w_string in words_to_find:
		w=[ord(ch)-96 for ch in list(w_string)]
		#Perform the search
		print("Seaching for " + w_string)
		lists_searched = []
		#Search row by row
		for r in range(0,GRID_LENGTH):
			lists_searched+=search(index_label_list(classified_mat,indices_grid[r]),w)
		#Search column by column
		for c in range(0,GRID_LENGTH):
			lists_searched+=search(index_label_list(classified_mat,indices_grid[:,c]),w)
		#Search all diagonals that are longer than or just as long as the word.
			lists_searched=trav_diag(classified_mat,indices_grid,lists_searched,w)

		#Likeliest is the tuple that has the biggest third element
		likeliest=sorted(lists_searched,key=lambda trituple: trituple[2])[len(lists_searched)-1]
		#Return the indices to start and end the line from.
		likeliest_tuple=likeliest[:2]
		#Display the result
		print("Line from index " + str(likeliest_tuple[0]) + " to index " + str(likeliest_tuple[1]))
		draw_line(plt,int(likeliest_tuple[0]), int(likeliest_tuple[1]), BLOCK_LENGTH)
	print("Solved")
	plt.gcf().savefig(get_name(testN), dpi=100)


wordsearch(test1, words, train_data, train_labels)
wordsearch(test2, words, train_data, train_labels)