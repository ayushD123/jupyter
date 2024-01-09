#1 A star
Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)],
}

# Define a function to get neighbors of a node 'v'
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None

# Define a heuristic function 'h' that provides estimated costs to reach the goal
def h(n):
    H_dist = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist[n]

# Define A* algorithm
def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)  # Initialize open set with start node
    closed_set = set()  # Initialize closed set

    g = {}  # Dictionary to store the cost from start to each node
    parents = {}  # Dictionary to store parent node of each node
    g[start_node] = 0  # Cost from start to start is 0
    parents[start_node] = start_node  # Parent of start node is itself

    # Loop until open set is not empty
    while len(open_set) > 0:
        n = None

        # Select node 'n' from open set with lowest total cost 'f = g + h'
        for v in open_set:
            if n == None or g[v] + h(v) < g[n] + h(n):
                n = v

        # Check if the current node 'n' is the goal node or has no neighbors
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            # Explore neighbors of current node 'n'
            for (m, weight) in get_neighbors(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)

        # Check if no valid node 'n' is found
        if n == None:
            print('Path does not exist!')
            return None
        # Check if goal node 'stop_node' is found
        if n == stop_node:
            path = []
            # Reconstruct and print the path from start to stop node
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path

        open_set.remove(n)  # Remove current node 'n' from open set
        closed_set.add(n)  # Add current node 'n' to closed set

    # If the loop ends without finding the goal node
    print('Path does not exist!')
    return None

# Execute A* algorithm from node 'A' to node 'J'
aStarAlgo('A', 'J')
o/p-
Path found: ['A', 'F', 'G', 'I', 'J']
['A', 'F', 'G', 'I', 'J']



3- candiadate-
import csv

with open("trainingexamples.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)

    specific = data[0][:-1]
    general = [['?' for i in range(len(specific))] for j in range(len(specific))]

    for i in data:
        if i[-1] == "Yes":
            for j in range(len(specific)):
                if i[j] != specific[j]:
                    specific[j] = "?"
                    general[j][j] = "?"

        elif i[-1] == "No":
            for j in range(len(specific)):
                if i[j] != specific[j]:
                    general[j][j] = specific[j]
                else:
                    general[j][j] = "?"

        print("\nStep " + str(data.index(i)+1) + " of Candidate Elimination Algorithm")
        print(specific)
        print(general)

    gh = [] # gh = general Hypothesis
    for i in general:
        for j in i:
            if j != '?':
                gh.append(i)
                break
    print("\nFinal Specific hypothesis:\n", specific)
    print("\nFinal General hypothesis:\n", gh)
o/p-
Step 1 of Candidate Elimination Algorithm
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Step 2 of Candidate Elimination Algorithm
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
[['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Step 3 of Candidate Elimination Algorithm
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]

Step 4 of Candidate Elimination Algorithm
['Sunny', 'Warm', '?', 'Strong', '?', '?']
[['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Final Specific hypothesis:
 ['Sunny', 'Warm', '?', 'Strong', '?', '?']

Final General hypothesis:
 [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]

4-id3-
# Import necessary libraries
import pandas as pd  # Import Pandas for data handling
from pprint import pprint  # Import pprint for pretty printing
from sklearn.feature_selection import mutual_info_classif  # Import mutual_info_classif for computing mutual information
from collections import Counter  # Import Counter for counting occurrences

# Define the ID3 algorithm function
def id3(df, target_attribute, attribute_names, default_class=None):
    # Count occurrences of classes in the target attribute
    cnt = Counter(x for x in df[target_attribute])

    # If all instances have the same class, return that class as a leaf node
    if len(cnt) == 1:
        return next(iter(cnt))

    # If the DataFrame is empty or no more attributes are left, return the default class
    elif df.empty or (not attribute_names):
        return default_class

    else:
        # Compute Information Gain (mutual information) for each attribute
        gains = mutual_info_classif(df[attribute_names], df[target_attribute], discrete_features=True)

        # Find the index of the attribute with the maximum gain
        index_of_max = gains.tolist().index(max(gains))
        best_attr = attribute_names[index_of_max]  # Get the attribute with the maximum gain
        tree = {best_attr: {}}  # Initialize a node for the best attribute
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]  # Remove the best attribute

        # Split the dataset based on the best attribute and build subtrees for each attribute value
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute, remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree  # Append subtrees under the best attribute in the tree

        return tree  # Return the constructed decision tree

# Read the dataset from a CSV file
df = pd.read_csv("traintennis_new.csv")

# Get a list of attribute names from the dataset columns
attribute_names = df.columns.tolist()

# Remove the target attribute ("PlayTennis") from the attribute list
attribute_names.remove("PlayTennis")

# Print the loaded dataset
print(df)

# Build the decision tree using the ID3 algorithm
tree = id3(df, "PlayTennis", attribute_names)

# Print a message indicating the display of the tree structure
print("The tree structure")

# Display the constructed decision tree structure in a human-readable format
pprint(tree)
o/p-
The tree structure
{'Outlook': {0: {'Humidity': {0: 0, 1: 1}}, 1: 1, 2: {'Wind': {0: 1, 1: 0}}}}






#5.ANN-
import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) # maximum of X array longitudinally
y = y/100

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 	#Setting training iterations
lr=0.1 		#Setting learning rate
inputlayer_neurons = 2 		#number of features in data set
hiddenlayer_neurons = 3 	#number of hidden layers neurons
output_neurons = 1 		#number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))  #2,3
bh=np.random.uniform(size=(1,hiddenlayer_neurons))                   #1,3
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))    #3,1
bout=np.random.uniform(size=(1,output_neurons))                      #1,1

for i in range(epoch):
#Forward Propogation 
    hinp=np.dot(X,wh)+ bh
    hlayer_act = sigmoid(hinp)      #HIDDEN LAYER ACTIVATION FUNCTION
    outinp=np.dot(hlayer_act,wout)+ bout
    output = sigmoid(outinp)

    outgrad = derivatives_sigmoid(output) 
    hiddengrad = derivatives_sigmoid(hlayer_act)
    
    EO = y-output                   #ERROR AT OUTPUT LAYER
    d_output = EO* outgrad

    EH = d_output.dot(wout.T)       #ERROR AT HIDDEN LAYER (TRANSPOSE => COZ REVERSE(BACK))
    d_hiddenlayer = EH * hiddengrad

    wout += hlayer_act.T.dot(d_output) *lr      #REMEMBER WOUT IS 3*1 MATRIX
    wh += X.T.dot(d_hiddenlayer) *lr

print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
o/p-
Input: 
[[0.66666667 1.        ]
 [0.33333333 0.55555556]
 [1.         0.66666667]]
Actual Output: 
[[0.92]
 [0.86]
 [0.89]]
Predicted Output: 
 [[0.89780745]
 [0.87606396]
 [0.89514409]]





8-KNN-
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import datasets
iris=datasets.load_iris() 
print("Iris Data set loaded...")
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
#random_state=0
for i in range(len(iris.target_names)):
    print("Label", i , "-",str(iris.target_names[i]))
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
print("Results of Classification using K-nn with K=5 ") 
for r in range(0,len(x_test)):
    print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r])," Predicted-label:", str(y_pred[r]))

    print("Classification Accuracy :" , classifier.score(x_test,y_test));
























9-Regression-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
y = np.log(np.abs((x ** 2) - 1) + 0.5)
x = x + np.random.normal(scale=0.05, size=1000) 
plt.scatter(x, y, alpha=0.3)
def local_regression(x0, x, y, tau): 
    x0 = np.r_[1, x0]
    x = np.c_[np.ones(len(x)), x]
    xw =x.T * radial_kernel(x0, x, tau) 
    beta = np.linalg.pinv(xw @ x) @ xw @ y 
    return x0 @ beta


def radial_kernel(x0, x, tau):
    return np.exp(np.sum((x - x0) ** 2, axis=1) / (-2 * tau ** 2))


def plot_lr(tau):
    domain = np.linspace(-5, 5, num=500)
    pred = [local_regression(x0, x, y, tau) for x0 in domain] 
    plt.scatter(x, y, alpha=0.3)
    plt.plot(domain, pred, color="red") 
    return plt


plot_lr(1).show()
