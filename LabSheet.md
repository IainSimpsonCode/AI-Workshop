# AI Workshop

In this lab, we will create 2 linear regression models: a simple linear regression model, and a multiple linear regression model.

## Simple Linear Regression

Simple linear regression works by looking at 1 feature vs 1 target. This is very basic, but will allow us to understand how these models work under-the-hood.

### Step 1: Setup

To start, we need to import some libraries:

```python
import numpy as np				# This will be used to handle our data

from numpy.random import seed	# These 2 will be used to generate our sample data
from numpy.random import randn

import matplotlib.pyplot as plt	# This will be used to plot our results and create graphs
```

And initialise some variables:

```python
c = 0
m = 0
# Remember these from our equation y = mx + c?
```



### Step 2: Create some sample data

After you've imported those, we will look at generating some sample data to work with. We need some linearly correlated data for this task, so we will create a function to generate some random points, with a general positive correlation.

```python
seed(10) # Change this value to get different test data

# Generate random x values
x = 20 * randn(1000) + 100
# Based on the x values, generate some y values
y = x + (10 * randn(1000) + 50)

# Show our test data
plt.scatter(x, y, alpha=0.6)
plt.show()
```

### Step 3: Calculate gradient

We are first going to find the gradient (m) of our line of best fit through all the points. This can be found using our equation:
<img width="373" height="142" alt="image" src="https://github.com/user-attachments/assets/796727a0-059b-45aa-bbed-73cc8a9da3e1" />

First, we need to find $\bar{x}$ and $\bar{y}$. Remember, this just means the average values of x and average values of y.
```python
avgX = np.mean(x)
avgY = np.mean(y)

print (avgX, avgY)
```

Now in our equation, we have a sum on the numerator (the top of the fraction), and a sum on the denominator (the bottom of the fraction). So we will work these out here:
```python
numeratorSum = 0
denominatorSum = 0

for i in range(len(x)):
    numeratorSum += (x[i] - avgX) * (y[i] - avgY)
    denominatorSum += (x[i] - avgX) ** 2

print(numeratorSum, denominatorSum)
```

Once we have worked out the sums on both sides of our fraction, we can divide this to find our gradient:
```python
m = numeratorSum / denominatorSum

print(m)
```

And see how it looks:
```python
# Plot the line of best fit so far with the correct gradient. Line has a y intercept of 0, which will be adjusted when c is calculated

plt.scatter(x, y, alpha=0.6)
plt.plot(x, m * x + c, color='red')
plt.show()
```
<img width="552" height="413" alt="image" src="https://github.com/user-attachments/assets/b7104aa5-a836-4f00-8118-fca97efd8d51" />

### Step 4: Find y-intercept
We can rearrange our equation $y = mx + c$, to get $c = y - mx$. Since we are working with the line of best fit, and not individual x and y values, we will use $\bar{x}$ and $\bar{y}$:
```python
c = avgY - (m * avgX)

print(c)
```

And see how that affects your line:
```python
plt.scatter(x, y, alpha=0.6)
plt.plot(x, m * x + c, color='red')
plt.show()
```
<img width="552" height="415" alt="image" src="https://github.com/user-attachments/assets/52a1c0a9-89db-45ae-a54b-5fa377376ccd" />
