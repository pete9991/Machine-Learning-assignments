# Report

**Name(s):**
**Student ID(s):**

---

## PART I: Logistic Regression

### Code

#### Summary and Results

- **In-sample accuracy:** …
- **Test accuracy:** …

**Generated Plot:**
![Logistic Regression Plot](logistic_plot.png)

_Comment (2 lines max):_
…

_Notes (if anything did not work):_
…

#### Actual Code

```python
# cost_grad
def cost_grad(...):
    ...

# fit
def fit(...):
    ...
```

---

### Theory

#### Q1. Running Time of Mini-Batch Gradient Descent

To find the running time of our metod we will split it in to its constituant parts, find their run time, and finally put it all together.
the method we are looking at here is the Fit method, which has the following variales:
n: training sampls
d: dimensions of the training data
E (epochs): the amount of epochs
B: batch size

the fit method can be broken down in to these parts:
create d-array (it takes d time)
the epoch loop (which runs E times)
permutation (runs n times)
shuffle x (takes nd time)
shuffle y (takes n time)
batch loop (runs n/d times)
x batch (takes Bd time)
y batch (takes B time)
cost-grad (takes nd+n+d time)
wehight update (takes constant time)
append result (takes same time as cost grad)

putting it all together we get (d+E(n+nd+n+n/d*(Bd+B+nd+n+d+1)+nd+n+d)) which will need to e reduced
(d+E(3n+2nd+d+n/d*(Bd+B+nd+n+d)))
(d+E(n+nd+d+(n*(Bd+B+nd+n+d))/d))
(d+E(n+nd+d+(n*(Bd+B+d))/d))
(d+E(n+nd+d+nd+n+(nd/B)))
(d+E(d+nd+n+(nd/B)))
(d+E(nd+(nd/B)))
(d+E(nd+nd)) worst case B=1 ergo (nd/B) = (nd/1) = nd
(d+E(nd))
(d+End)
(End)
the O time of our method is O(End)
…

#### Q2. Sanity Check (Cats vs. Dogs)

If we apply the same fixed permutation to all the images, it reorders the feature indices for every element. If we also use the same reordering for the weights to correspond to the features, then the decision function is mathematically unchanged, given it is the same permutation at test. The performance will therefore be the same.
Mathematically, we can let P denote a random permutation matrix. Then we have $$ (Pw)^T (Px) = w^T(P^TP)x = w^Tx $$ since $P^TP = I$.

#### Q3. Linearly Separable Data

Given the data is linearly separable and logistic regression is implemented with gradient descent, then every time gradient descent is run, it will reduce the loss. In other words:
$$ log(1+e^{(-c\cdot \text{a positive number})}) \rightarrow 0 $$ when $$ c \rightarrow \infty $$
in other words, by letting $||w||$ converge to $\infty$ then we can minimize the loss as much as possible. There is therefor no finite w that minimizes. Infimum is 0, but achieved when the limit $||w|| \rightarrow 0 $

---

## PART II: Softmax Regression

### Code

#### Summary and Results

- **Wine data set – In-sample accuracy:** …

- **Wine data set – Test accuracy:** …

- **MNIST data set – In-sample accuracy:** …

- **MNIST data set – Test accuracy:** …

**Generated Plots:**
![Softmax Wine Plot](softmax_wine.png)
![Softmax Digits Plot](softmax_digits.png)
![Softmax Digits Visualization](softmax_visualization.png)

_Comment (2 lines max):_
…

#### Actual Code

```python
# cost_grad
def cost_grad(...):
    ...

# fit
def fit(...):
    ...
```

---

### Theory

#### Q1. Running Time of Softmax Implementation

here we'll be doing the same thig as in question 1, just on the cost grad method using soft max
n: training sampls
d: dimensions of the training data
k: classes

one-in-k-notation (takes nk time)
soft_max (takes nd-time)
np.arrange (takes n time)
np.mean (takes n time)
grad calculations (takes (dnk+nk+dk) -> dnk time)

putting it all together we get O(nk+nd+n+n+dnk) which we will then reduce
nk+nd+n+dnk
dnk (since nk, nd and n are all part of dnk we can elminate them)
the time for cost grad is O(dnk)
…

---

# Appendix (Optional)

- Extra plots
- Additional tests
