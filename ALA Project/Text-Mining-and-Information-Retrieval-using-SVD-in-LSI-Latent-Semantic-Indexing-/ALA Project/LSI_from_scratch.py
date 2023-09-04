import pandas as pd
import math
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from string import punctuation
import nltk
import os

pi = math.pi
path = str(os.getcwd()) + "\\"
inp = [
    path + 'Test1.txt',
    path + 'Test2.txt',
    path + 'Test3.txt',
    path + 'Test4.txt',
    path + 'Test5.txt',
    path + 'Test6.txt',
]


def transpose(A):
    ans = np.zeros((len(A[0]), len(A)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            ans[j][i] = A[i][j]
    return ans


def matrix_multiplication(A, X):
    if type(X) != list and len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    ans = np.zeros((len(A), len(X[0])))
    for i in range(len(A)):
        for j in range(len(X[0])):
            for k in range(len(X)):
                ans[i][j] = ans[i][j] + A[i][k] * X[k][j]
    if type(X) != list and ans.shape[1] == 1:
        ans = ans.reshape((len(ans)))
    return ans


def SVD(A_Matrix):
    Original_Matrix = A_Matrix.copy()

    rows = len(A_Matrix)
    cols = len(A_Matrix[0])

    if rows > cols:
        A_Matrix = transpose(A_Matrix)
    B = A_Matrix.copy()

    A_Matrix = matrix_multiplication(transpose(A_Matrix), A_Matrix)

    # HERE A_Matrix IS CONVERTED TO SQUARE MATRIX THUS rows=cols
    rows = len(A_Matrix)
    cols = len(A_Matrix[0])

    A_Matrix = np.round(A_Matrix, decimals=5)
    Diagonal_Matrix = A_Matrix.copy()
    Eigen_Vectors_Matrix = np.zeros((rows, rows))
    for i in range(0, rows):
        Eigen_Vectors_Matrix[i][i] = 1

    max = 1
    # Loop is iterated untill the max element does not become 0.
    while max > 0.001:
        # ith_Max_Element is the ith index of the max element other then diagonal
        # jth_Max_Element is the jth index of the max element other then diagonal

        max, ith_Max_Element, jth_Max_Element = math.fabs(Diagonal_Matrix[0][1]), 0, 1
        for i in range(0, len(Diagonal_Matrix)):
            for j in range(0, len(Diagonal_Matrix[0])):
                if i != j and math.fabs(Diagonal_Matrix[i][j]) > max:
                    max = math.fabs(Diagonal_Matrix[i][j])
                    ith_Max_Element = i
                    jth_Max_Element = j
        theta = 0

        if (
            Diagonal_Matrix[ith_Max_Element][ith_Max_Element]
            == Diagonal_Matrix[jth_Max_Element][jth_Max_Element]
        ):
            if Diagonal_Matrix[ith_Max_Element][jth_Max_Element] > 0:
                theta = pi / 4
            else:
                theta = -1 * pi / 4
        else:
            value = (
                2
                * Diagonal_Matrix[ith_Max_Element][jth_Max_Element]
                / (
                    Diagonal_Matrix[ith_Max_Element][ith_Max_Element]
                    - Diagonal_Matrix[jth_Max_Element][jth_Max_Element]
                )
            )
            theta = abs(0.5 * math.atan(value))
        Orthogonal_Matrix = np.zeros((len(Diagonal_Matrix), len(Diagonal_Matrix[0])))

        for i in range(0, len(Diagonal_Matrix)):
            Orthogonal_Matrix[i][i] = 1
        Orthogonal_Matrix[ith_Max_Element][ith_Max_Element] = math.cos(theta)
        Orthogonal_Matrix[jth_Max_Element][jth_Max_Element] = Orthogonal_Matrix[
            ith_Max_Element
        ][ith_Max_Element]
        Orthogonal_Matrix[ith_Max_Element][jth_Max_Element] = math.sin(theta)
        Orthogonal_Matrix[jth_Max_Element][ith_Max_Element] = (
            -1 * Orthogonal_Matrix[ith_Max_Element][jth_Max_Element]
        )

        # Diagonal_Matrix= P^(-1)*D*P     where P^(-1) is P transpose

        Diagonal_Matrix = matrix_multiplication(
            transpose(Orthogonal_Matrix), Diagonal_Matrix
        )

        Diagonal_Matrix = matrix_multiplication(Diagonal_Matrix, Orthogonal_Matrix)

        Eigen_Vectors_Matrix = matrix_multiplication(
            Eigen_Vectors_Matrix, Orthogonal_Matrix
        )

    # l1 is the list of eigen values which is extracted from the diagonal of the Diagonal_Matrix.
    l1 = []

    for i in range(0, len(Diagonal_Matrix)):
        l1.append(Diagonal_Matrix[i][i])

    # Eigen_Vectors_Matrix is a list of eigenvectors -- VT
    Eigen_Vectors_Matrix = transpose(Eigen_Vectors_Matrix)

    temp_List = list(Eigen_Vectors_Matrix.copy())

    for i in range(0, len(temp_List)):
        temp_List[i] = list(temp_List[i])
        temp_List[i].insert(0, l1[i])

    # Sorting the eigen values in descending order and simuntaneously arranging the eigen vectors w.r.t descending eigen values.
    temp_List.sort(reverse=True)
    Eigen_Values_List = []
    Eigen_Finalvectors_List = []
    for i in range(0, len(temp_List)):
        Eigen_Values_List.append(temp_List[i].pop(0))
        Eigen_Finalvectors_List.append(temp_List[i])
    Eigen_Values_Array = np.array(Eigen_Values_List)
    # VT is ready now
    Eigen_Finalvectors_Transpose_Array = np.array(Eigen_Finalvectors_List)

    # Initiallizing U matrix
    U_Matrix = np.zeros((len(B), len(B)))
    for i in range(len(B)):
        # Ui=(B*Vi)/root(lamda)
        mul = matrix_multiplication((B), (Eigen_Finalvectors_Transpose_Array[i]))

        U_Matrix[i] = mul / math.sqrt(Eigen_Values_Array[i])
    U_Matrix = transpose(U_Matrix)
    Final_Eigen_Values = []
    for i in Eigen_Values_Array:
        if i > 1e-4:  # we check if i!=0
            Final_Eigen_Values.append(i)
    # Singular values are root of every eigen values.
    Final_Singular_Values = [math.sqrt(i) for i in Final_Eigen_Values]
    Final_Singular_Values = np.round(Final_Singular_Values, decimals=6)

    # Initiallizing Sigma matrix
    Sigma = np.zeros((len(B), len(B[0])))
    for i in range(0, min(len(B), len(B[0]))):
        Sigma[i][i] = Final_Singular_Values[i]
    VT = Eigen_Finalvectors_Transpose_Array

    if len(Original_Matrix) > len(Original_Matrix[0]):
        # For rows>columns we had taken Atranspose above and we had found U*Sigma*VT of AT
        # Thus this will be equivalent to V*SigmaT*UT of A.
        Sigma = transpose(Sigma)
        U_Temp = U_Matrix.copy()
        U_Matrix = transpose(Eigen_Finalvectors_Transpose_Array)
        VT = transpose(U_Temp)
    return U_Matrix, Final_Singular_Values, VT


docs = len(inp)  # number of documents
store = []  # list of content of all docs. stored as a string
for i in range(len(inp)):
    file = open(inp[i], "r")
    string = ""
    while 1:
        # read by character
        char = file.read(1)
        if char != "\n":
            string += char
        if not char:
            break
    store.append(string)
    file.close

nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")


def rem_punct(s):
    """
    Parameter:
        s - string
    Process:
        First we initialize an empty output string str1.

        The function iterates through s and if the character is not a punctation,
        it is added to the outpur string str1.
    Output:
        str1 - string
    """
    str1 = ""
    for char in s:
        if char not in punctuation:
            str1 = str1 + char
    return str1


def rem_nums(s):
    """
    Parameter:
        s - string
    Process:
        First we initialize an empty output string str1.

        The function iterates through s and if the character is not a number,
        it is added to the outpur string str1.
    Output:
        str1 - string
    """
    str1 = ""
    for num in s:
        if not num.isdigit():
            str1 = str1 + num
    return str1


all_docs = []
for string in store:
    string = rem_punct(rem_nums(string).lower())
    string = string.split(" ")
    all_docs.append(string)

bow = []  # bag of words
for j in range(docs):
    temp = []
    for i in all_docs[j]:  # all_docs[j] is the jth document's list of words
        if i not in stopwords and len(i) > 0 and i != None:
            temp.append(i)
    bow.append(temp)


def unique(bow):
    """
    Parameter:
        bow - 2D list
    Process:
        a is initialized as first list of bow

        The function iterates through all other lists and sets
        a = union(a,cur_list)
        Here cur_list is the iterator
    Output:
        a - set of all unique words in bow
    """
    a = bow[0]
    for i in range(1, len(bow)):
        a = set(a).union(set(bow[i]))
    return a


wordset = unique(bow)
worddict = [dict.fromkeys(wordset, 0) for i in range(len(bow))]


def term_document_matrix():
    """
    Parameter:
        Nothing
    Process:
        Iterates through bow and worddict at the same time

        bow - 2D list  (iterator is bow_i)
        worddict = list of dictionaries    (iterator is worddict_i)

        increments value of each word found in bow_i
    Output:
        pandas dataframe
    """
    for bow_i, worddict_i in zip(
        bow, worddict
    ):  # zip takes first row of bow and key of worddict
        for word in bow_i:
            worddict_i[
                word
            ] += 1  # increments value of each word when found in the document

    return pd.DataFrame(worddict)


# The above function returns a document term matrix.
# from this we get number of times a unique word is found in each document respectively.

docterm = term_document_matrix()


def term_freq(worddict, bow):
    """
    Parameter:
        worddict - dictionary
        bow - list of strings
    Process:
        initialize empty dictionary called tfdict
        bowcount = number of elements(words) in bow

        traverse through key value pairs of worddict
        Here,
            word - key
            count - value
        Divide count of a word by total number of elements in that document and store it as a
        key value pair in tfdict
    Output:
        tfdict - dictionary
    """
    # here worddict is a single dictionary. NOT A LIST OF DICTIONARIES
    # bow is a single list. NOT A NESTED LIST
    tfdict = {}  # tfdict -- term frequency dict
    bowcount = len(bow)  # bowcount = total number of words in the document
    for word, count in worddict.items():
        tfdict[word] = count / float(bowcount)
    return tfdict


tfbow = []
for i, j in zip(worddict, bow):  # worddict is a list of dictionaries
    tfbow.append(term_freq(i, j))

# tfbow is a list of dictionaries. ith dictionary in tfbow is the tfdict of ith document


def idf(doclist):
    """
    Parameters:
        doclist - list of dictionaries
    Process:
        returns a dictionary containing key value pairs of words and number of documents
        that words occurs in.
        The process is described in detail below.
    Output:
        idfdict - dictionary
    """
    idfdict = {}
    n = len(doclist)
    idfdict = dict.fromkeys(doclist[0].keys(), 0)
    # initializes idfdict as a dictionary which has same keys as doclist[0] and value of each key is 0
    for doc in doclist:
        for word, val in doc.items():
            if val > 0:
                idfdict[word] += 1
    # now idfdict has total occurences of each word in all of the documents
    # note that multiple occurences of a word in one document is considered as 1
    # hence if value of a key is... for eg: {'king':2,....}
    # this means that the word king has appeared in 2 documents
    for word, val in idfdict.items():
        idfdict[word] = math.log(n / float(val))
        # computes log (total num of documents/no. of documents that contain a particular word)n
    return idfdict


idfs = idf(worddict)  # stores idf value of all words


def tfidf(tfbow_dict, idfs):
    """
    Parameter:
        tfbow_dict - dictionary
        idfs - dictionary
    Process:
        multiplies term frequency with idf of each term
    Output:
        tfidf - dictionary
    """
    tfidf = {}
    for word, val in tfbow_dict.items():
        tfidf[word] = val * idfs[word]  # multiply term freq with idf for each term
    return tfidf


tfidfl = []
for i in tfbow:
    tfidfl.append(tfidf(i, idfs))
# Above loop iterates through list of dictionaries tfbow.
# Appends output of tfidf function(a dictionary) to tfidfl
X = pd.DataFrame(tfidfl).T
l_2d = X.T.values.tolist()
l_2d = np.array(l_2d)
l_2d = transpose(l_2d)
L, S, R = SVD(l_2d)


def zero_padding(n, U, S, V):
    """
    Parameter:
        n - integer
        U - Matrix having dimension m*m
        S - List of singular values
        V - Matrix having dimension n*n
    Process:
        The aim of this function is to use S to make a matrix A which contains n singular
        values on its diagonal. Rest all values are 0. Dimensions of A are m*n
        Basically, we want to make A such that multiply(U,A,V) is possible
    Output:
        A - Matrix having dimension m*n
    """
    t = []
    for i in range(0, n):
        t.append(S[i])
    # t is the tuple consisting of first n characters of the tuple S (S is sigma)
    for i in range(len(S) - n):
        t.append(0)  # zero_padding t with zeros

    # convert the non-selected singular values to 0 and form a diagonal matrix, store as A
    A = []
    for i in range(0, len(S)):
        temp = []
        for j in range(0, len(S)):
            if j == i:
                temp.append(float(t[j]))
            else:
                temp.append(float(0))
        A.append(temp)
    newrow = []
    for i in range(0, len(S)):
        newrow.append(0)
    # newrow is a list of zeros having same length as S
    for i in range(len(U) - len(S)):
        A.append(newrow)

    return A


def reconstruct(u, s, v, n):
    """
    Parameter:
        u - Matrix having dimension m*m
        s - List of singular values
        v - Matrix having dimension n*n
        n - integer
    Process:
        produces matrix A having m*n dimensions.
        Returns the product of multiply(u,A,v)
    Output:
        m*n matrix
    """
    A = zero_padding(n, u, s, v)
    return np.round(matrix_multiplication(matrix_multiplication(u, A), v), decimals=3)


def frobenius(a, a2):  # finds how similar two matrices are
    """
    Parameter:
        a - Matrix having dimension m*n
        a2 - Matrix having dimension m*n
    Process:
        computes the frobenius norm of the matrix
    Output:
        an integer (frobenius norm of the two matrices)
    """

    tempi1 = (a - a2) ** 2
    tempi2 = a**2
    sum1, sum2 = 0, 0

    for i in tempi1:
        for j in i:
            sum1 += j
    for i in tempi2:
        for j in i:
            sum2 += j
    ret = math.sqrt(sum1) / math.sqrt(sum2)
    return ret


def find_k():
    """
    Parameter:
        None
    Process:
        iterates through 1 to number of singular values - 1.
        compares the two matrices l_2d and reconstructed matrix of l_2d

        Note that l_2d is the original matrix (the one passed into SVD)
    Output:
        an integer - the number of singular values required to reconstruct a matrix
                     whose frobenius norm with l_2d is less than 0.38
    """
    for i in range(1, len(S)):
        f = frobenius(l_2d, reconstruct(L, S, R, i))
        if f < 0.35:
            return i


def search(q):
    q = rem_punct(q)
    q = q.lower().split(" ")
    terms = X.index

    query = []
    for i in terms:
        if i in q:
            query.append(1)
        else:
            query.append(0)
    query = np.asarray(query)
    if 1 not in query:
        print("Could not find any documents")

    else:
        k = find_k()
        reconstructed_A = reconstruct(L, S, R, k)
        score = query.dot(reconstructed_A)

        sort = {}
        for i in range(len(score)):
            if score[i] < 0:
                score[i] = 0
            sort[i + 1] = score[i]
        last = {}
        for w in sorted(sort, key=sort.get, reverse=True):
            last[w] = sort[w]
        xAxis = []
        title = "Document-wise weightage of the string : "
        for i in q:
            title = title + " " + i
        yAxis = []

        for document, sc in last.items():
            print("Document: ", document)
            xAxis.append(document)
            yAxis.append(sc)

        plt.bar(xAxis, yAxis, color="red", width=0.4)

        plt.xlabel("Document number")
        plt.ylabel("Relevance-Score")
        plt.title(title)
        plt.show()


while 1:
    print(
        "Please choose an option:\n"
        + "1: Search keywords \n"
        + "2: View Documents \n"
        + "3: Exit"
    )
    opt = int(input())
    if opt == 1:
        print("Enter keyword you want to search\n")
        s = input()
        print("Loading your search results...")
        search(s)
        print("\n")
    elif opt == 2:
        print("Enter Document number :\n\n")
        doc_n = int(input())

        if doc_n > len(store):
            print("There are only ", len(store), " documents")
            continue
        print(store[doc_n - 1])
        print("\n")
    elif opt == 3:
        print("Thank You")
        exit()
    else:
        print("Enter valid input. ")
