# EAI320_Practical-6
# Implemented By: Mohamed Ameen Omar
# EAI 320 - Intelligent Systems, University of Pretoria

## Practical Assignment 6

### Compiled by Dr J. Dabrowski

### Edited and reviewed by Johan Langenhoven

### 08 April 2018


## Question 1

Implement the backpropagation algorithm for artificial neural networks (ANNs) as a
Python function. The function should perform backpropagation on a three layered
network withD inputs,H hidden neurons andC outputs. Both the input layer and
the output layer should contain a bias neuron. The general form of this neural network
is illustrated in the figure below:

The function should take the following input arguments.

- The input data as aD×N array, whereDis the number of features andNis the
    number of data samples. (Note thatDincludes the bias neuron).
- The target values as aC×N array, whereCis the number of output classes.
- The initial values of the weights from the input layer to the hidden layer as a
    D×(H−1) array.His the number of neurons in the hidden layer, including the
    bias neuron.
- The initial values of the weights from the hidden layer to the output layer as a
    H×Carray.
- The learning rate,η.
- The terminating conditions: number of epochs, and error convergence.


The function should output the following parameters.

- The updated input-to-hidden layer weight array
- The updated hidden-to-output layer weight array.
- A vector of error values for each epoch (see discussion below.)

In each epoch (training iteration), the function should compute the error:

```
E=
```
#### 1

#### 2

#### ∑

```
k
```
```
(tk−zk)^2
```
The values should be stored in a vector with a length equal to that of the total number
of epochs. The error convergence criterion can be calculated as the absolute value of
the difference between the current error and the previous error. When this reaches a
specified threshold the algorithm terminates. The activation function and its derivative
should be implemented in separate python functions. Alwaysensure that the dataset
features are normalized to the range [0,1] before passing them to the backpropagation
algorithm.

## Question 2

In this question you are required to demonstrate the ability of a ANN to model complex
decision boundaries. Apply the backpropagation algorithm developed in Question 1 to
model an arrow shaped decision boundary. For this, a dataset is created from the low
resolution image illustrated below.

The input dataset consists of horizontal and vertical pixel indices. The target dataset
consists of binary values of the pixel value corresponding to each pixel index. The


input and target datasets are contained in the comma delimited files ‘q2inputs.csv’ and
‘q2targets.csv’ respectively. Remember to normalise the inputs.

Plot a three dimensional figure containing the two features on the horizontal axes and
the class probability (ANN output) on the vertical axis. The following code may be
used to plot such a figure:

#Generate the surface
i = 0
j = 0
Z = np.zeros ((21 ,21))
for x1 in np.arange (0 ,1 ,0.05):
i = 0
for x2 in np.arange (0 ,1 ,0.05):
#Calculate the output of the neural network
#given the input [1 x1 x2 ] and ANN weights , W1 and W
Z [i, j] = ffNeuralNet (np.array ([[1] ,[x1],[x2]] , W1, W2)
i = i+
j = j+
#Plot the surface
X = np.arange (0,21,1)
Y = np.arange (0,21,1)
X, Y = np.meshgrid(X, Y)
fig = plt.figure( )
ax = fig.gca(projection = '3d')
surf = ax.plotsurface(X,Y,Z, rstride = 1, cstride = 1, cmap=cm.
coolwarm)
ax.setzlim (0,1)

## Question 3

Apply the ANN to classify a wine dataset. The dataset consists of 13 features describing
a wine sample. These features are listed as:


1. Alcohol
2. Malic Acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total Phenols
7. Flavanoids
8. Nonflavanoid Phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

Each sample from the dataset comes from one of three different cultivars in Italy. The
objective is to train an ANN to be able to determine from which cultivar a wine sample
originates, given the 13 features extracted from the wine sample. The dataset has been
split into a training and test set. The task is to train the ANN on the training set and
then test it using the test set. The training set inputs and target data are contained
in the ‘q3TrainInputs.csv’ and ‘q3TrainTargets.csv’ files respectively. The test set in-
puts and target data are contained in the ‘q3TestInputs.csv’ and ‘q3TestTargets.csv’
files respectively. Note that the inputs have been normalised.

Include the following in your results:

1. Increment the number of hidden layer neurons from 1 until zero errors occur.
2. Plot the error over the training epochs.

## Report

You have to write a short technical report for this assignment. Your report must be
written in LATEX. In the report you will give your results as well as provide a discussion
on the results. Make sure to follow the guidelines as set out in the separate questions
to form a complete report.

Reports will only be handed in as digital copies, but a hard copy plagiarism statement
needs to be handed in at the following week’s practical session (on the final day of the
practical submission).


## Deliverable

- Write a technical report on your finding for this assignment.
- Include your code in the digital submission as an appendix.

## Instructions

- All reports must be in PDF format and be named report.pdf.
- Place the software in a folder called SOFTWARE and the report in a folder called
    REPORT.
- Add the folders to a zip-archive and name itEAI320_prac6_studnr.zip.
- All reports and simulation software must be e-mailed toup.eai320@gmail.com
    no later than 17:00 on 20 April 2018. No late submissions will be accepted.
- Use the following format for the subject header for the email: EAI 320 Prac 6 -
    studnr.
- Bring your plagiarism statements to the practical session on Thursday, 12 April
    2018, where they will be collected.
- Submit your report online on ClickUP using the TurnItIn link.

## Additional Instructions

- Do not copy! The copier and the copyee (of software and/or documentation) will
    receive zero for both the software and the documentation.
- For any questions or appointments email me atup.eai320@gmail.com.
- Make sure that you discuss the results that are obtained. This is a large part of
    writing a technical report.

## Marking

Your report will be marked as follow:

- 60% will be awarded for the full implementation of the practical and the subsequent
    results in the report. For partially completed practicals, marks will be awarded as
    seen fit by the marker.Commented code allows for easier marking!
- 40% will be awarded for the overall report. This includes everything from the
    report structure, grammar and discussion of results. The discussion will be the
    bulk of the marks awarded.


