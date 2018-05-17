:::::::::::::::::::::::::::::::::
:: Checkerboard Datasets ::::::::
:::::::::::::::::::::::::::::::::

A non-Gaussian data set is derived from the canonical XOR problem, which 
resembles a rotating checkerboard. The rotation makes this deceptively
simple-looking problem particularly challenging, as the angle and location 
of the decision boundaries change rather drastically at each time step. 
After half a rotation, data are drawn from a recurring environment, as 
the [pi 2*pi] interval will create an identical distribution drift to that 
of the [0 pi] interval.

We introduce four variations of this dataset to observe a learner’s 
resilience in the presence of harsher environments with varying drift 
rate, that is, where the rate of change in the distribution is not 
constant. This is accomplished by applying positive or negative 
acceleration to the alpha (rotation) parameter as it increases from 
0 to 2*pi. A constant drift rate is tested along with an exponentially 
increasing, pulsing, or sinusoidally fluctuating drift rate. 


::: Naming Convention ::::::::::::::::::::::::::::::::::::::::

Four checkerboard datasets are available: 
 "CBconstant": checkerboard data with constant drift rate 
 "CBpulsing": checkerboard data with pulsing drift rate
 "CBexponential": checkerboard data with exponentially increasing drift rate
 "CBsinusoidal": checkerboard data with sinusoidal drift rate


::: Dataset Information ::::::::::::::::::::::::::::::::::::::

 Features: 2
 Classes: 2
 Time steps: 400
 Training instances per time step: 25
 Testing instances per time step: 1,024
 Total Training Instances: 10,000
 Total Testing Instances: 409,600


::: Training/Testing Procedure (Batch Learning) ::::::::::::::
 
 At each time step, read in 25 training samples from 
 "training_data.csv" (10,000 x 2) with corresponding class labels  from
 "training_class.csv" (10,000 x 1 vector denoting class "1" or class "2"), 
 and 1024 testing samples from "testing_data.csv" (409,600 x 2) with 
 corresponding class labels from "testing_class.csv" (409,600 x 1).  
 For subsequent time steps, shift the training and testing windows by 25 
 and 1,024, respectively, and read in the next batch.



==========================================================
Ryan Elwell, Robi Polikar
Signal Processing & Pattern Recognition Laboratory (SPPRL)
Department of Electrical & Computer Engineering
Rowan University
==========================================================